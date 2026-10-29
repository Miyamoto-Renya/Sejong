use libloading::Library;
use sejong_core::{Error, Result, ensure};
use std::{
    cell::RefCell,
    ffi::{CStr, CString, c_char, c_int, c_uint, c_void},
    path::Path,
    rc::Rc,
};

type Handle = *mut c_void;
macro_rules! hip_api {
    ($($field:ident: $name:literal => $ty:ty),* $(,)?) => {
        pub(crate) struct Hip { _library: Library, $(pub $field: $ty,)* }
        impl Hip {
            pub fn load(path: &Path) -> Result<Self> {
                unsafe {
                    let library = Library::new(path).map_err(|e| Error::Message(format!("HIP library {}: {e}",path.display())))?;
                    $(let $field = *library.get::<$ty>(concat!($name,"\0").as_bytes()).map_err(|e| Error::Message(e.to_string()))?;)*
                    Ok(Self { _library: library, $($field,)* })
                }
            }
        }
    }
}
hip_api! {
    init: "hipInit" => unsafe extern "C" fn(c_uint)->c_int,
    set_device: "hipSetDevice" => unsafe extern "C" fn(c_int)->c_int,
    device_name: "hipDeviceGetName" => unsafe extern "C" fn(*mut c_char,c_int,c_int)->c_int,
    mem_info: "hipMemGetInfo" => unsafe extern "C" fn(*mut usize,*mut usize)->c_int,
    malloc: "hipMalloc" => unsafe extern "C" fn(*mut Handle,usize)->c_int,
    free: "hipFree" => unsafe extern "C" fn(Handle)->c_int,
    memcpy: "hipMemcpy" => unsafe extern "C" fn(Handle,*const c_void,usize,c_int)->c_int,
    stream_create: "hipStreamCreate" => unsafe extern "C" fn(*mut Handle)->c_int,
    stream_sync: "hipStreamSynchronize" => unsafe extern "C" fn(Handle)->c_int,
    stream_destroy: "hipStreamDestroy" => unsafe extern "C" fn(Handle)->c_int,
    module_load: "hipModuleLoadData" => unsafe extern "C" fn(*mut Handle,*const c_void)->c_int,
    module_function: "hipModuleGetFunction" => unsafe extern "C" fn(*mut Handle,Handle,*const c_char)->c_int,
    module_unload: "hipModuleUnload" => unsafe extern "C" fn(Handle)->c_int,
    launch: "hipModuleLaunchKernel" => unsafe extern "C" fn(Handle,c_uint,c_uint,c_uint,c_uint,c_uint,c_uint,c_uint,Handle,*mut Handle,*mut Handle)->c_int,
    error_string: "hipGetErrorString" => unsafe extern "C" fn(c_int)->*const c_char,
}
impl Hip {
    pub fn check(&self, code: c_int, operation: &str) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        let message = unsafe { CStr::from_ptr((self.error_string)(code)) }.to_string_lossy();
        Err(Error::Message(format!(
            "{operation}: HIP {code}: {message}"
        )))
    }
}

pub(crate) struct DeviceInner {
    pub hip: Hip,
    pub stream: Handle,
    pub name: String,
    budget: usize,
    pool: RefCell<Vec<(Handle, usize)>>,
    allocated: RefCell<usize>,
    peak: RefCell<usize>,
}
impl DeviceInner {
    pub fn new(library: &Path, device: i32, budget: usize) -> Result<Rc<Self>> {
        let hip = Hip::load(library)?;
        let mut stream = std::ptr::null_mut();
        let mut name = [0i8; 256];
        unsafe {
            hip.check((hip.init)(0), "hipInit")?;
            hip.check((hip.set_device)(device), "hipSetDevice")?;
            hip.check(
                (hip.device_name)(name.as_mut_ptr(), 256, device),
                "hipDeviceGetName",
            )?;
            hip.check((hip.stream_create)(&mut stream), "hipStreamCreate")?;
        }
        let mut available = 0;
        let mut total = 0;
        unsafe {
            hip.check((hip.mem_info)(&mut available, &mut total), "hipMemGetInfo")?;
        }
        let budget = if budget == 0 {
            available
        } else {
            budget.min(available)
        };
        ensure(budget > 0, "GPU has no available allocation capacity")?;
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        Ok(Rc::new(Self {
            hip,
            stream,
            name,
            budget,
            pool: RefCell::new(vec![]),
            allocated: RefCell::new(0),
            peak: RefCell::new(0),
        }))
    }
    pub fn budget(&self) -> usize {
        self.budget
    }
    pub fn available(&self) -> Result<usize> {
        self.synchronize()?;
        self.reclaim()?;
        let mut available = 0;
        let mut total = 0;
        unsafe {
            self.hip.check(
                (self.hip.mem_info)(&mut available, &mut total),
                "hipMemGetInfo",
            )?;
        }
        Ok(available.min(self.budget.saturating_sub(*self.allocated.borrow())))
    }
    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            self.hip
                .check((self.hip.stream_sync)(self.stream), "hipStreamSynchronize")
        }
    }
    pub fn allocate(self: &Rc<Self>, bytes: usize) -> Result<Rc<Allocation>> {
        let size = crate::Engine::allocation_bytes(bytes)?;
        let found = self.pool.borrow().iter().position(|(_, n)| *n == size);
        if let Some(index) = found {
            let (pointer, bytes) = self.pool.borrow_mut().swap_remove(index);
            return Ok(Rc::new(Allocation {
                device: self.clone(),
                pointer,
                bytes,
            }));
        }
        if self.allocated.borrow().saturating_add(size) > self.budget {
            self.reclaim()?;
        }
        ensure(
            self.allocated.borrow().saturating_add(size) <= self.budget,
            format!(
                "GPU memory budget exceeded: allocated={} MiB, request={} MiB, budget={} MiB",
                *self.allocated.borrow() / 1048576,
                size / 1048576,
                self.budget / 1048576
            ),
        )?;
        let mut pointer = std::ptr::null_mut();
        unsafe {
            self.hip
                .check((self.hip.malloc)(&mut pointer, size), "hipMalloc")?;
        }
        *self.allocated.borrow_mut() += size;
        let peak = (*self.peak.borrow()).max(*self.allocated.borrow());
        *self.peak.borrow_mut() = peak;
        Ok(Rc::new(Allocation {
            device: self.clone(),
            pointer,
            bytes: size,
        }))
    }
    pub fn reclaim(&self) -> Result<()> {
        self.synchronize()?;
        for (pointer, bytes) in self.pool.borrow_mut().drain(..) {
            unsafe {
                self.hip.check((self.hip.free)(pointer), "hipFree")?;
            }
            *self.allocated.borrow_mut() -= bytes;
        }
        Ok(())
    }
    pub fn peak(&self) -> usize {
        *self.peak.borrow()
    }
}
impl Drop for DeviceInner {
    fn drop(&mut self) {
        unsafe {
            (self.hip.stream_sync)(self.stream);
            for (pointer, _) in self.pool.get_mut().drain(..) {
                (self.hip.free)(pointer);
            }
            (self.hip.stream_destroy)(self.stream);
        }
    }
}
pub(crate) struct Allocation {
    device: Rc<DeviceInner>,
    pub pointer: Handle,
    bytes: usize,
}
impl Drop for Allocation {
    fn drop(&mut self) {
        self.device
            .pool
            .borrow_mut()
            .push((self.pointer, self.bytes));
    }
}

pub(crate) struct Module {
    device: Rc<DeviceInner>,
    module: Handle,
    pub function: Handle,
    pub shared: u32,
    pub threads: u32,
    pub global_scratch: usize,
    pub profile_scratch: usize,
}
impl Module {
    pub fn load(
        device: Rc<DeviceInner>,
        binary: &Path,
        symbol: &str,
        shared: u32,
        threads: u32,
        global_scratch: usize,
        profile_scratch: usize,
    ) -> Result<Self> {
        let data = std::fs::read(binary)?;
        let mut module = std::ptr::null_mut();
        let mut function = std::ptr::null_mut();
        let name = CString::new(symbol).map_err(|e| Error::Message(e.to_string()))?;
        unsafe {
            device.hip.check(
                (device.hip.module_load)(&mut module, data.as_ptr().cast()),
                "hipModuleLoadData",
            )?;
            if let Err(e) = device.hip.check(
                (device.hip.module_function)(&mut function, module, name.as_ptr()),
                "hipModuleGetFunction",
            ) {
                (device.hip.module_unload)(module);
                return Err(e);
            }
        }
        Ok(Self {
            device,
            module,
            function,
            shared,
            threads,
            global_scratch,
            profile_scratch,
        })
    }
}
impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            (self.device.hip.stream_sync)(self.device.stream);
            (self.device.hip.module_unload)(self.module);
        }
    }
}
