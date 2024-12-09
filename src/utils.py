from hest import STReader, VisiumReader, VisiumHDReader, XeniumReader


# st_type = ['ST', 'Visium', 'Xenium']
def load_st(path, platform):
    assert platform in ['st', 'visium', 'visium-hd', 'xenium'], "platform must be one of ['st', 'visium', 'visium-hd', 'xenium']"
    
    if platform == 'st':    
        st = STReader().auto_read(path)
        
    if platform == 'visium':
        st = VisiumReader().auto_read(path)
        
    if platform == 'visium-hd':
        st = VisiumHDReader().auto_read(path)
        
    if platform == 'xenium':
        st = XeniumReader().auto_read(path)
        
    return st

