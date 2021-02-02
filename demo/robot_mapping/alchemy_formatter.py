from demo.robot_mapping.robot_map_loader import process_data, load_raw_data, load_predicate_data, merge_processed_data
from utils import load, save, save_str_list


def load_data_fold(fold):
    map_names = ['a', 'l', 'n', 'u', 'w']
    raw_maps = [load_raw_data(f'radish.rm.raw/{map_name}.map', map_name) for map_name in map_names]
    predicate_maps = [load_predicate_data(f'radish.rm/{map_name}.db', map_name) for map_name in map_names]
    processed_maps = [process_data(raw_maps[i], predicate_maps[i]) for i in range(5)]
    alchemy_maps = [to_alchemy_format(raw_maps[i], predicate_maps[i], processed_maps[i]) for i in range(5)]

    train = list()
    for i in range(5):
        if i != fold:
            train.extend(alchemy_maps[i])

    return train, alchemy_maps[fold]

def to_alchemy_format(raw_maps, predicate_maps, processed_maps):
    res = list()

    for args in predicate_maps['Aligned']:
        res.append(f'Aligned({",".join(args)})')
    for args in predicate_maps['Neighbors']:
        res.append(f'Neighbors({",".join(args)})')
    for args in predicate_maps['Sequence']:
        res.append(f'Consecutive({",".join(args)})')
    for args in predicate_maps['SmoothSharpTurn']:
        res.append(f'SharpTurn({",".join(args)})')

    type_dict = {'W': 'Wall', 'D': 'Door', 'O': 'Other'}
    for arg, value in processed_maps['seg_type'].items():
        res.append(f'SegType({arg},{type_dict[value]})')
    for arg, value in processed_maps['length'].items():
        res.append(f'Length({arg}) {value}')
    for arg, value in processed_maps['depth'].items():
        res.append(f'Depth({arg}) {value}')
    for arg, value in processed_maps['angle'].items():
        res.append(f'Angle({arg}) {value}')

    return res


if __name__ == '__main__':
    for i in range(5):
        train, test = load_data_fold(1)
        save_str_list(f'alchemy/{i}/train', train)
        save_str_list(f'alchemy/{i}/test', test)
