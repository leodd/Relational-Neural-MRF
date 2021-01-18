save_cache = False


def train_mod(is_train):
    global save_cache
    if is_train:
        save_cache = True
    else:
        save_cache = False
