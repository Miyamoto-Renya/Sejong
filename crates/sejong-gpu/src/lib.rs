//! Rust-owned HIP tensors and native launch of Triton-compiled kernels.
mod common_ops;
pub mod graph;
mod hip;
mod parser_ops;
use hip::{Allocation, DeviceInner, Module};
use sejong_core::{DType, Error, GpuConfig, Result, ensure, json};
use std::{
    collections::HashMap,
    ffi::c_void,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    rc::Rc,
};

#[derive(Clone)]
pub struct Tensor {
    allocation: Rc<Allocation>,
    offset: usize,
    pub shape: Vec<usize>,
    pub dtype: DType,
}
impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .finish()
    }
}
impl Tensor {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn pointer(&self) -> u64 {
        self.allocation.pointer as u64 + self.offset as u64
    }
    pub fn rows(&self, start: usize, count: usize) -> Result<Self> {
        ensure(
            self.shape.len() == 2
                && count > 0
                && start
                    .checked_add(count)
                    .is_some_and(|end| end <= self.shape[0]),
            "row slice out of bounds",
        )?;
        Ok(Self {
            allocation: self.allocation.clone(),
            offset: self.offset + start * self.shape[1] * self.dtype.bytes(),
            shape: vec![count, self.shape[1]],
            dtype: self.dtype,
        })
    }
    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        ensure(
            shape.iter().product::<usize>() == self.len(),
            "reshape size mismatch",
        )?;
        Ok(Self {
            allocation: self.allocation.clone(),
            offset: self.offset,
            shape: shape.to_vec(),
            dtype: self.dtype,
        })
    }
}
struct Compiler {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}
impl Compiler {
    fn start(root: &Path) -> Result<Self> {
        let python = std::env::var_os("SEJONG_PYTHON")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join(".venv/bin/python"));
        let mut child = Command::new(python)
            .args(["-m", "sejong_kernels.bridge.compiler"])
            .current_dir(root)
            .env("SEJONG_KERNEL_CACHE", root.join(".cache/kernels"))
            .env("TRITON_CACHE_DIR", root.join(".cache/triton"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        Ok(Self {
            input: child.stdin.take().unwrap(),
            output: BufReader::new(child.stdout.take().unwrap()),
            child,
        })
    }
    fn compile(&mut self, request: &json::Value) -> Result<json::Value> {
        serde_write(&mut self.input, request)?;
        self.input.flush()?;
        let mut line = String::new();
        ensure(
            self.output.read_line(&mut line)? > 0,
            "Triton compiler exited before returning a kernel",
        )?;
        let response: json::Value = json::from_str(&line)?;
        if let Some(error) = response.get("error") {
            return Err(Error::Message(format!("Triton compiler: {error}")));
        }
        response
            .get("ok")
            .cloned()
            .ok_or_else(|| Error::Message("Triton compiler response has no result".into()))
    }
}
fn serde_write(writer: &mut impl Write, value: &json::Value) -> Result<()> {
    writer.write_all(json::to_string(value)?.as_bytes())?;
    writer.write_all(b"\n")?;
    Ok(())
}
impl Drop for Compiler {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct Engine {
    device: Rc<DeviceInner>,
    compiler: Compiler,
    kernels: HashMap<String, Module>,
    config: GpuConfig,
}
#[derive(Clone)]
pub enum Arg {
    Ptr(Tensor),
    I32(i32),
    F32(f32),
}
impl Arg {
    fn signature(&self) -> &str {
        match self {
            Self::Ptr(t) => t.dtype.pointer(),
            Self::I32(_) => "i32",
            Self::F32(_) => "fp32",
        }
    }
    fn value(&self) -> u64 {
        match self {
            Self::Ptr(t) => t.pointer(),
            Self::I32(v) => *v as u32 as u64,
            Self::F32(v) => v.to_bits() as u64,
        }
    }
}
fn p(t: &Tensor) -> Arg {
    Arg::Ptr(t.clone())
}
fn n(v: usize) -> Result<Arg> {
    Ok(Arg::I32(i32::try_from(v).map_err(|_| {
        Error::Message("kernel dimension exceeds i32".into())
    })?))
}
fn f(v: f32) -> Arg {
    Arg::F32(v)
}
impl Engine {
    pub fn new(config: GpuConfig) -> Result<Self> {
        let root = sejong_core::root();
        let library = if let Some(path) = std::env::var_os("SEJONG_HIP_LIBRARY") {
            PathBuf::from(path)
        } else if let Some(path) = std::env::var_os("ROCM_PATH") {
            PathBuf::from(path).join("lib/libamdhip64.so.7")
        } else {
            let output=Command::new(root.join(".venv/bin/python")).args(["-c","import importlib.metadata as m; print(m.distribution('rocm-sdk-core').locate_file('_rocm_sdk_core/lib/libamdhip64.so.7'))"]).output()?;
            ensure(
                output.status.success(),
                format!(
                    "ROCm discovery: {}",
                    String::from_utf8_lossy(&output.stderr)
                ),
            )?;
            PathBuf::from(String::from_utf8_lossy(&output.stdout).trim())
        };
        let device = DeviceInner::new(
            &library,
            config.device,
            config
                .memory_mb
                .checked_mul(1048576)
                .ok_or_else(|| Error::Message("GPU memory setting overflow".into()))?,
        )?;
        Ok(Self {
            device,
            compiler: Compiler::start(&root)?,
            kernels: HashMap::new(),
            config,
        })
    }
    pub fn allocation_bytes(bytes: usize) -> Result<usize> {
        bytes
            .max(1)
            .checked_next_power_of_two()
            .ok_or_else(|| Error::Message("GPU allocation size overflow".into()))
    }
    pub fn memory_budget_bytes(&self) -> usize {
        self.device.budget()
    }
    pub fn available_memory_bytes(&self) -> Result<usize> {
        self.device.available()
    }
    pub fn context_id(&self) -> usize {
        std::rc::Rc::as_ptr(&self.device) as usize
    }
    pub fn name(&self) -> &str {
        &self.device.name
    }
    pub fn synchronize(&self) -> Result<()> {
        self.device.synchronize()
    }
    pub fn reclaim(&self) -> Result<()> {
        self.device.reclaim()
    }
    pub fn peak_bytes(&self) -> usize {
        self.device.peak()
    }
    pub fn empty(&self, shape: &[usize], dtype: DType) -> Result<Tensor> {
        let count = shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| Error::Message("tensor dimensions overflow".into()))?;
        ensure(
            count > 0 && count <= i32::MAX as usize,
            "tensor size must be in 1..=i32::MAX",
        )?;
        Ok(Tensor {
            allocation: self.device.allocate(count * dtype.bytes())?,
            offset: 0,
            shape: shape.to_vec(),
            dtype,
        })
    }
    fn upload(&self, shape: &[usize], dtype: DType, data: &[u8]) -> Result<Tensor> {
        let t = self.empty(shape, dtype)?;
        ensure(
            data.len() == t.len() * dtype.bytes(),
            "host tensor size mismatch",
        )?;
        self.synchronize()?;
        unsafe {
            self.device.hip.check(
                (self.device.hip.memcpy)(
                    t.pointer() as *mut c_void,
                    data.as_ptr().cast(),
                    data.len(),
                    1,
                ),
                "host-to-device copy",
            )?;
        }
        Ok(t)
    }
    pub fn from_f32(&self, shape: &[usize], data: &[f32]) -> Result<Tensor> {
        self.upload(shape, DType::F32, unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast(), data.len() * 4)
        })
    }
    pub fn from_f64(&self, shape: &[usize], data: &[f64]) -> Result<Tensor> {
        self.upload(shape, DType::F64, unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast(), data.len() * 8)
        })
    }
    pub fn to_f64(&mut self, t: &Tensor) -> Result<Vec<f64>> {
        let t = self.cast(t, DType::F64)?;
        self.synchronize()?;
        let mut data = vec![0f64; t.len()];
        unsafe {
            self.device.hip.check(
                (self.device.hip.memcpy)(
                    data.as_mut_ptr().cast(),
                    t.pointer() as *const c_void,
                    data.len() * 8,
                    2,
                ),
                "device-to-host f64",
            )?;
        }
        Ok(data)
    }
    pub fn from_i32(&self, shape: &[usize], data: &[i32]) -> Result<Tensor> {
        self.upload(shape, DType::I32, unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast(), data.len() * 4)
        })
    }
    pub fn to_f32(&mut self, t: &Tensor) -> Result<Vec<f32>> {
        let t = self.cast(t, DType::F32)?;
        self.synchronize()?;
        let mut data = vec![0f32; t.len()];
        unsafe {
            self.device.hip.check(
                (self.device.hip.memcpy)(
                    data.as_mut_ptr().cast(),
                    t.pointer() as *const c_void,
                    data.len() * 4,
                    2,
                ),
                "device-to-host copy",
            )?;
        }
        Ok(data)
    }
    pub fn launch(
        &mut self,
        operation: &str,
        args: &[(&str, Arg)],
        constants: json::Value,
        grid: [usize; 3],
    ) -> Result<()> {
        ensure(
            grid.iter().all(|v| *v > 0 && *v <= u32::MAX as usize),
            "invalid kernel grid",
        )?;
        let signature: json::Map<String, json::Value> = args
            .iter()
            .map(|(name, arg)| {
                (
                    name.to_string(),
                    json::Value::String(arg.signature().into()),
                )
            })
            .collect();
        let request = json::json!({"operation":operation,"signature":signature,"constants":constants,"architecture":self.config.architecture,"wavefront":self.config.wavefront,"warps":4});
        let key = json::to_string(&request)?;
        if !self.kernels.contains_key(&key) {
            let r = self.compiler.compile(&request)?;
            let module = Module::load(
                self.device.clone(),
                Path::new(
                    r["binary"]
                        .as_str()
                        .ok_or_else(|| Error::Message("missing kernel binary".into()))?,
                ),
                r["symbol"].as_str().unwrap(),
                r["shared"].as_u64().unwrap() as u32,
                r["threads"].as_u64().unwrap() as u32,
                r["global_scratch"].as_u64().unwrap() as usize,
                r["profile_scratch"].as_u64().unwrap() as usize,
            )?;
            self.kernels.insert(key.clone(), module);
        }
        let kernel = &self.kernels[&key];
        let programs = grid.iter().product::<usize>();
        let global = if kernel.global_scratch > 0 {
            Some(self.device.allocate(kernel.global_scratch * programs)?)
        } else {
            None
        };
        let profile = if kernel.profile_scratch > 0 {
            Some(self.device.allocate(kernel.profile_scratch * programs)?)
        } else {
            None
        };
        let mut values: Vec<u64> = args.iter().map(|(_, arg)| arg.value()).collect();
        values.push(global.as_ref().map_or(0, |v| v.pointer as u64));
        values.push(profile.as_ref().map_or(0, |v| v.pointer as u64));
        let mut pointers: Vec<*mut c_void> =
            values.iter_mut().map(|v| (v as *mut u64).cast()).collect();
        unsafe {
            self.device.hip.check(
                (self.device.hip.launch)(
                    kernel.function,
                    grid[0] as u32,
                    grid[1] as u32,
                    grid[2] as u32,
                    kernel.threads,
                    1,
                    1,
                    kernel.shared,
                    self.device.stream,
                    pointers.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                operation,
            )?;
        }
        Ok(())
    }
    pub fn full(&mut self, shape: &[usize], dtype: DType, value: f32) -> Result<Tensor> {
        let y = self.empty(shape, dtype)?;
        self.launch(
            "fill",
            &[("Y", p(&y)), ("n", n(y.len())?), ("value", f(value))],
            json::json!({"BLOCK":256}),
            [y.len().div_ceil(256), 1, 1],
        )?;
        Ok(y)
    }
    pub fn cast(&mut self, x: &Tensor, dtype: DType) -> Result<Tensor> {
        if x.dtype == dtype {
            return Ok(x.clone());
        }
        let y = self.empty(&x.shape, dtype)?;
        self.copy_into(x, &y)?;
        Ok(y)
    }
    pub fn copy_into(&mut self, x: &Tensor, y: &Tensor) -> Result<()> {
        ensure(x.len() == y.len(), "copy shape mismatch")?;
        self.launch(
            "copy",
            &[("X", p(x)), ("Y", p(y)), ("n", n(x.len())?)],
            json::json!({"BLOCK":256}),
            [x.len().div_ceil(256), 1, 1],
        )
    }
    pub fn add(&mut self, x: &Tensor, z: &Tensor, alpha: f32, dtype: DType) -> Result<Tensor> {
        ensure(x.shape == z.shape, "add shape mismatch")?;
        let y = self.empty(&x.shape, dtype)?;
        self.add_into(x, z, &y, alpha)?;
        Ok(y)
    }
    pub fn add_into(&mut self, x: &Tensor, z: &Tensor, y: &Tensor, alpha: f32) -> Result<()> {
        ensure(
            x.shape == z.shape && x.shape == y.shape,
            "add shape mismatch",
        )?;
        self.launch(
            "add",
            &[
                ("X", p(x)),
                ("Z", p(z)),
                ("Y", p(y)),
                ("n", n(x.len())?),
                ("alpha", f(alpha)),
            ],
            json::json!({"BLOCK":256}),
            [x.len().div_ceil(256), 1, 1],
        )
    }
    pub fn matmul(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        ta: bool,
        tb: bool,
        dtype: DType,
    ) -> Result<Tensor> {
        ensure(
            a.shape.len() == 2 && b.shape.len() == 2,
            "matmul requires matrices",
        )?;
        let (m, k) = if ta {
            (a.shape[1], a.shape[0])
        } else {
            (a.shape[0], a.shape[1])
        };
        let (kb, nn) = if tb {
            (b.shape[1], b.shape[0])
        } else {
            (b.shape[0], b.shape[1])
        };
        ensure(k == kb, "matmul inner dimension mismatch")?;
        let c = self.empty(&[m, nn], dtype)?;
        if dtype == DType::F64 {
            self.launch(
                "matmul64",
                &[
                    ("A", p(a)),
                    ("B", p(b)),
                    ("C", p(&c)),
                    ("M", n(m)?),
                    ("N", n(nn)?),
                    ("K", n(k)?),
                ],
                json::json!({"TA":ta,"TB":tb,"BLOCK":32}),
                [m, nn.div_ceil(32), 1],
            )?;
            return Ok(c);
        }

        self.launch(
            "matmul",
            &[
                ("A", p(a)),
                ("B", p(b)),
                ("C", p(&c)),
                ("M", n(m)?),
                ("N", n(nn)?),
                ("K", n(k)?),
            ],
            json::json!({"TA":ta,"TB":tb,"BM":16,"BN":32,"BK":32}),
            [m.div_ceil(16), nn.div_ceil(32), 1],
        )?;
        Ok(c)
    }
    pub fn embedding(&mut self, w: &Tensor, ids: &Tensor, dtype: DType) -> Result<Tensor> {
        let d = w.shape[1];
        let y = self.empty(&[ids.len(), d], dtype)?;
        self.launch(
            "embedding",
            &[
                ("W", p(w)),
                ("Ids", p(ids)),
                ("Y", p(&y)),
                ("N", n(ids.len())?),
                ("D", n(d)?),
            ],
            json::json!({"BLOCK":256}),
            [y.len().div_ceil(256), 1, 1],
        )?;
        Ok(y)
    }
    pub fn embedding_backward(
        &mut self,
        dy: &Tensor,
        ids: &Tensor,
        wshape: &[usize],
    ) -> Result<Tensor> {
        let dw = self.full(wshape, DType::F32, 0.)?;
        self.launch(
            "embedding_backward",
            &[
                ("DY", p(dy)),
                ("Ids", p(ids)),
                ("DW", p(&dw)),
                ("N", n(ids.len())?),
                ("D", n(wshape[1])?),
            ],
            json::json!({"BLOCK":256}),
            [dy.len().div_ceil(256), 1, 1],
        )?;
        Ok(dw)
    }
    pub fn norm(&mut self, x: &Tensor, w: &Tensor, epsilon: f32) -> Result<Tensor> {
        let d = x.shape[1];
        ensure(w.len() == d, "RMSNorm weight mismatch")?;
        let y = self.empty(&x.shape, x.dtype)?;
        self.launch(
            "rmsnorm",
            &[("X", p(x)), ("W", p(w)), ("Y", p(&y))],
            json::json!({"D":d,"EPS":epsilon,"BLOCK":d.next_power_of_two()}),
            [x.shape[0], 1, 1],
        )?;
        Ok(y)
    }
    pub fn norm_backward(
        &mut self,
        x: &Tensor,
        w: &Tensor,
        dy: &Tensor,
        epsilon: f32,
    ) -> Result<(Tensor, Tensor)> {
        let d = x.shape[1];
        let dx = self.empty(&x.shape, DType::F32)?;
        let dw = self.full(&w.shape, DType::F32, 0.)?;
        self.launch(
            "rmsnorm_backward",
            &[
                ("X", p(x)),
                ("W", p(w)),
                ("DY", p(dy)),
                ("DX", p(&dx)),
                ("DW", p(&dw)),
            ],
            json::json!({"D":d,"EPS":epsilon,"BLOCK":d.next_power_of_two()}),
            [x.shape[0], 1, 1],
        )?;
        Ok((dx, dw))
    }
    pub fn swiglu(&mut self, g: &Tensor, u: &Tensor) -> Result<Tensor> {
        ensure(g.shape == u.shape, "SwiGLU shape mismatch")?;
        let y = self.empty(&g.shape, g.dtype)?;
        self.launch(
            "swiglu",
            &[("G", p(g)), ("U", p(u)), ("Y", p(&y)), ("n", n(g.len())?)],
            json::json!({"BLOCK":256}),
            [g.len().div_ceil(256), 1, 1],
        )?;
        Ok(y)
    }
    pub fn swiglu_backward(
        &mut self,
        g: &Tensor,
        u: &Tensor,
        dy: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let dg = self.empty(&g.shape, DType::F32)?;
        let du = self.empty(&u.shape, DType::F32)?;
        self.launch(
            "swiglu_backward",
            &[
                ("G", p(g)),
                ("U", p(u)),
                ("DY", p(dy)),
                ("DG", p(&dg)),
                ("DU", p(&du)),
                ("n", n(g.len())?),
            ],
            json::json!({"BLOCK":256}),
            [g.len().div_ceil(256), 1, 1],
        )?;
        Ok((dg, du))
    }
    pub fn rope(
        &mut self,
        x: &Tensor,
        t: usize,
        offset: usize,
        heads: usize,
        theta: f32,
        inverse: bool,
    ) -> Result<Tensor> {
        let d = x.shape[1] / heads;
        let y = self.empty(&x.shape, x.dtype)?;
        self.launch(
            "rope",
            &[
                ("X", p(x)),
                ("Y", p(&y)),
                ("N", n(x.len())?),
                ("T", n(t)?),
                ("OFFSET", n(offset)?),
            ],
            json::json!({"D":d,"H":heads,"THETA":theta,"INVERSE":inverse,"BLOCK":256}),
            [(x.len() / 2).div_ceil(256), 1, 1],
        )?;
        Ok(y)
    }
    pub fn attention(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        tq: usize,
        tk: usize,
        offset: usize,
        heads: usize,
    ) -> Result<(Tensor, Tensor)> {
        self.attention_mode(q, k, v, tq, tk, offset, heads, true)
    }
    pub fn attention_mode(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        tq: usize,
        tk: usize,
        offset: usize,
        heads: usize,
        causal: bool,
    ) -> Result<(Tensor, Tensor)> {
        ensure(
            k.shape == v.shape
                && q.shape[1] == k.shape[1]
                && q.shape[0] / tq == k.shape[0] / tk
                && offset + tq <= tk,
            "attention shape/position mismatch",
        )?;
        let d = q.shape[1] / heads;
        let y = self.empty(&q.shape, q.dtype)?;
        let lse = self.empty(
            &[q.shape[0] * heads],
            if q.dtype == DType::F64 {
                DType::F64
            } else {
                DType::F32
            },
        )?;
        self.launch(
            "attention",
            &[
                ("Q", p(q)),
                ("K", p(k)),
                ("V", p(v)),
                ("O", p(&y)),
                ("LSE", p(&lse)),
                ("TQ", n(tq)?),
                ("TK", n(tk)?),
                ("OFFSET", n(offset)?),
            ],
            json::json!({"H":heads,"D":d,"BD":d.next_power_of_two(),"BK":32,"CAUSAL":causal,"FP64":q.dtype==DType::F64}),
            [q.shape[0] * heads, 1, 1],
        )?;
        Ok((y, lse))
    }
    pub fn attention_backward(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        o: &Tensor,
        lse: &Tensor,
        dy: &Tensor,
        t: usize,
        heads: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        self.attention_backward_mode(q, k, v, o, lse, dy, t, heads, true)
    }
    pub fn attention_backward_mode(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        o: &Tensor,
        lse: &Tensor,
        dy: &Tensor,
        t: usize,
        heads: usize,
        causal: bool,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let d = q.shape[1] / heads;
        let dq = self.empty(&q.shape, DType::F32)?;
        let dk = self.empty(&k.shape, DType::F32)?;
        let dv = self.empty(&v.shape, DType::F32)?;
        let constants =
            json::json!({"H":heads,"D":d,"BD":d.next_power_of_two(),"BK":32,"CAUSAL":causal});
        let grid = [q.shape[0] * heads, 1, 1];
        self.launch(
            "attention_dq",
            &[
                ("Q", p(q)),
                ("K", p(k)),
                ("V", p(v)),
                ("O", p(o)),
                ("LSE", p(lse)),
                ("DO", p(dy)),
                ("DQ", p(&dq)),
                ("T", n(t)?),
            ],
            constants.clone(),
            grid,
        )?;
        self.launch(
            "attention_dkv",
            &[
                ("Q", p(q)),
                ("K", p(k)),
                ("V", p(v)),
                ("O", p(o)),
                ("LSE", p(lse)),
                ("DO", p(dy)),
                ("DK", p(&dk)),
                ("DV", p(&dv)),
                ("T", n(t)?),
            ],
            constants,
            grid,
        )?;
        Ok((dq, dk, dv))
    }
    pub fn cross_entropy(&mut self, x: &Tensor, targets: &Tensor) -> Result<Tensor> {
        ensure(x.shape[0] == targets.len(), "cross entropy row mismatch")?;
        let v = x.shape[1];
        let loss = self.empty(&[x.shape[0]], DType::F32)?;
        self.launch(
            "cross_entropy",
            &[("X", p(x)), ("Targets", p(targets)), ("Loss", p(&loss))],
            json::json!({"V":v,"BLOCK":v.next_power_of_two()}),
            [x.shape[0], 1, 1],
        )?;
        Ok(loss)
    }
    pub fn cross_entropy_backward(
        &mut self,
        x: &Tensor,
        targets: &Tensor,
        dy: &Tensor,
    ) -> Result<Tensor> {
        let v = x.shape[1];
        let dx = self.empty(&x.shape, DType::F32)?;
        self.launch(
            "cross_entropy_backward",
            &[
                ("X", p(x)),
                ("Targets", p(targets)),
                ("DY", p(dy)),
                ("DX", p(&dx)),
            ],
            json::json!({"V":v,"BLOCK":v.next_power_of_two()}),
            [x.shape[0], 1, 1],
        )?;
        Ok(dx)
    }
    pub fn squared_norm(&mut self, x: &Tensor) -> Result<f64> {
        let blocks = x.len().div_ceil(256);
        let y = self.empty(&[blocks], DType::F32)?;
        self.launch(
            "sum_squares",
            &[("X", p(x)), ("Y", p(&y)), ("N", n(x.len())?)],
            json::json!({"BLOCK":256}),
            [blocks, 1, 1],
        )?;
        Ok(self.to_f32(&y)?.into_iter().map(f64::from).sum())
    }
    pub fn adamw(
        &mut self,
        w: &Tensor,
        g: &Tensor,
        m: &Tensor,
        v: &Tensor,
        step: usize,
        c: &sejong_core::TrainConfig,
        lr: f32,
        scale: f32,
        decay: bool,
    ) -> Result<()> {
        self.launch(
            "adamw",
            &[
                ("W", p(w)),
                ("G", p(g)),
                ("M", p(m)),
                ("V", p(v)),
                ("N", n(w.len())?),
                ("LR", f(lr)),
                ("BETA1", f(c.beta1)),
                ("BETA2", f(c.beta2)),
                ("EPS", f(c.epsilon)),
                ("DECAY", f(if decay { c.weight_decay } else { 0. })),
                ("INV_B1", f(1. / (1. - c.beta1.powf(step as f32)))),
                ("INV_B2", f(1. / (1. - c.beta2.powf(step as f32)))),
                ("SCALE", f(scale)),
            ],
            json::json!({"BLOCK":256}),
            [w.len().div_ceil(256), 1, 1],
        )
    }
}

impl Engine {
    pub fn select_rows(&mut self, a: &Tensor, b: &Tensor, mask: &Tensor) -> Result<Tensor> {
        ensure(
            a.shape == b.shape && a.shape.len() == 2 && mask.len() == a.shape[0],
            "row selection shape mismatch",
        )?;
        let y = self.empty(&a.shape, a.dtype)?;
        self.launch(
            "select_rows",
            &[
                ("A", p(a)),
                ("B", p(b)),
                ("Mask", p(mask)),
                ("Y", p(&y)),
                ("N", n(a.len())?),
                ("D", n(a.shape[1])?),
            ],
            json::json!({"BLOCK":256}),
            [a.len().div_ceil(256), 1, 1],
        )?;
        Ok(y)
    }
}

impl Engine {
    pub fn gelu(&mut self, x: &Tensor) -> Result<Tensor> {
        let y = self.empty(&x.shape, x.dtype)?;
        self.launch(
            "gelu",
            &[("X", p(x)), ("Y", p(&y)), ("N", n(x.len())?)],
            json::json!({"BLOCK":256}),
            [x.len().div_ceil(256), 1, 1],
        )?;
        Ok(y)
    }
    pub fn gelu_backward(&mut self, x: &Tensor, dy: &Tensor) -> Result<Tensor> {
        let y = self.empty(&x.shape, DType::F32)?;
        self.launch(
            "gelu_backward",
            &[
                ("X", p(x)),
                ("DY", p(dy)),
                ("DX", p(&y)),
                ("N", n(x.len())?),
            ],
            json::json!({"BLOCK":256}),
            [x.len().div_ceil(256), 1, 1],
        )?;
        Ok(y)
    }
    pub fn gather(&mut self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        ensure(
            x.shape.len() == 2 && ids.dtype == DType::I32,
            "gather requires matrix and row indices",
        )?;
        let y = self.empty(&[ids.len(), x.shape[1]], x.dtype)?;
        self.launch(
            "gather",
            &[
                ("X", p(x)),
                ("Ids", p(ids)),
                ("Y", p(&y)),
                ("N", n(ids.len())?),
                ("D", n(x.shape[1])?),
            ],
            json::json!({"BLOCK":256}),
            [y.len().div_ceil(256), 1, 1],
        )?;
        Ok(y)
    }
    pub fn gather_backward(
        &mut self,
        dy: &Tensor,
        ids: &Tensor,
        shape: &[usize],
    ) -> Result<Tensor> {
        let dx = self.full(shape, DType::F32, 0.)?;
        self.launch(
            "gather_backward",
            &[
                ("DY", p(dy)),
                ("Ids", p(ids)),
                ("DX", p(&dx)),
                ("N", n(ids.len())?),
                ("D", n(shape[1])?),
            ],
            json::json!({"BLOCK":256}),
            [dy.len().div_ceil(256), 1, 1],
        )?;
        Ok(dx)
    }
}
