import traceback

def typc(obj :any, comp :any) :
    return type(obj) == type(comp)

def error(e, msg :str ="", cry :bool =True, ex :bool =False) :
    print("ERROR :", msg, e)
    if cry : traceback.print_exc()
    if ex  : exit()