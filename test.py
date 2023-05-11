import psutil
import ctypes
import win32process
import win32api
import win32con

def get_base_address(pid):
    process_all_access = (0x000F0000 | 0x00100000 | 0xFFF)
    process_handle = win32api.OpenProcess(process_all_access, False, pid)

    modules = win32process.EnumProcessModules(process_handle)
    process_base_address = modules[0]

    win32api.CloseHandle(process_handle)
    return process_base_address

def read_memory(process_handle, address):
    buffer = ctypes.c_ulonglong()
    bytesRead = ctypes.c_ulonglong()
    ctypes.windll.kernel32.ReadProcessMemory(process_handle, ctypes.c_void_p(address), ctypes.byref(buffer), 8, ctypes.byref(bytesRead))
    return buffer.value

def get_process_id(process_name):
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == process_name:
            return proc.info['pid']
    return None

def get_memory_value(pid, base_address, offsets):
    PROCESS_VM_READ = 0x10
    process_handle = ctypes.windll.kernel32.OpenProcess(PROCESS_VM_READ, False, pid)

    address = base_address
    for offset in offsets[:-1]:
        address = address + offset
        address = read_memory(process_handle, address)

    address += offsets[-1]
    value = read_memory(process_handle, address)

    ctypes.windll.kernel32.CloseHandle(process_handle)
    return address, value

process_name = "mame.exe"
offsets = [0x137423E8, 0x250, 0x18, 0x8ACF]

pid = get_process_id(process_name)
if pid:
    print(f"Process ID of {process_name} is {pid}")
    base_address = get_base_address(pid)
    print(f"Base address: {hex(base_address)}")

    final_address, value = get_memory_value(pid, base_address, offsets)
    print(f"The value at address {hex(final_address)} is {value}")
else:
    print(f"{process_name} not found")
