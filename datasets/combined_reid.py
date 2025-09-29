import itertools
from collections import defaultdict
from .bases import BaseImageDataset
from .nightreid import NightReID
from .syndark import SynDark
from .cuhk import CUHK
from .msmt_cycle_gan import MSMT17
from .market1501 import Market1501


FACTORY = {
    'syndark': SynDark,
    'nightreid': NightReID,
    'cuhk': CUHK,
    'msmt': MSMT17,
    'market': Market1501
}
def _pid_range(items):
    if not items: return None
    ps = [p for _, p, _, _ in items]
    return min(ps), max(ps)

class CombinedReID(BaseImageDataset):
    def __init__(self,
                 specs=None,
                 root='',
                 eval_mode='last',
                 llie='',
                 eval_dataset=None,
                 verbose=True,
                 ds_kwargs=None):
        super().__init__()
        if specs is None:
            specs = [
                'market',
                'msmt',
                'cuhk',
                'nightreid'
            ]
        ds_kwargs = ds_kwargs or {}

        self.specs = specs
        self.eval_mode = eval_mode
        self.eval_dataset = (eval_dataset.lower() if isinstance(eval_dataset, str) else None)

        pid_offset = 0
        cam_offset = 0

        loaded = []  # [(name_lower, ds_obj, n_used_pids, n_used_cams)]
        trains, queries, galleries = [], [], []

        for name in self.specs:
            key = name.lower()
            if key not in FACTORY:
                raise KeyError(f"Unknown dataset name '{name}'. FACTORY keys: {list(FACTORY.keys())}")

            DS = FACTORY[key]
            ds_root = root

            # 데이터셋 로드: 현재까지의 pid_offset / cam_offset 적용
            if llie != '' and name == 'nightreid':
                ds = DS(root=ds_root,
                        verbose=False,
                        pid_begin=pid_offset,
                        cam_begin=cam_offset,
                        llie=llie,
                        **ds_kwargs)
            else:
                ds = DS(root=ds_root,
                    verbose=False,
                    pid_begin=pid_offset,
                    cam_begin=cam_offset,
                    **ds_kwargs)
            
            
            # === 디버그 ===
            print(f"[{name}] passed pid_begin={pid_offset}, cam_begin={cam_offset}")
            print(f"[{name}] TRAIN pid range = {_pid_range(ds.train)}  (len={len(ds.train)})")
            if len(ds.train) > 0:
                print(f"[{name}] TRAIN first item = {ds.train[0]}")
            # ==============
            
            train_used_pids = {pid for _, pid, _, _ in ds.train}
            train_used_cams = {cam for _, _, cam, _ in ds.train}
            n_train_used_pids = len(train_used_pids)
            n_train_used_cams = len(train_used_cams)

            if verbose:
                all_used_pids = {pid for _, pid, _, _ in itertools.chain(ds.train, ds.query, ds.gallery)}
                all_used_cams = {cam for _, _, cam, _ in itertools.chain(ds.train, ds.query, ds.gallery)}
                print(
                    f"[{name}] loaded: train={len(ds.train)}, query={len(ds.query)}, gallery={len(ds.gallery)} | "
                    f"train_unique_pids={n_train_used_pids}, train_unique_cams={n_train_used_cams} | "
                    f"all_unique_pids={len(all_used_pids)}, all_unique_cams={len(all_used_cams)} | "
                    f"applied pid_begin={pid_offset}, cam_begin={cam_offset}"
                )

            # 다음 데이터셋을 위해 오프셋 증가
            pid_offset += n_train_used_pids
            cam_offset += n_train_used_cams

            loaded.append((key, ds, n_train_used_pids, n_train_used_cams))

            # 학습 세트는 모두 합침
            trains += ds.train
            # 평가 세트 합치는지는 아래 eval_mode에서 처리

        # 평가 세트 구성
        if self.eval_mode == 'combine':
            # 모든 데이터셋의 query/gallery 합치기
            for _, ds, _, _ in loaded:
                queries += ds.query
                galleries += ds.gallery
        elif self.eval_mode == 'dataset':
            if not self.eval_dataset:
                raise ValueError("eval_mode='dataset' 인 경우 eval_dataset 이름을 지정해야 합니다.")
            picked = [ds for key, ds, _, _ in loaded if key == self.eval_dataset]
            if not picked:
                raise ValueError(f"eval_dataset='{self.eval_dataset}' 를 specs에서 찾을 수 없습니다.")
            ds = picked[0]
            queries = ds.query
            galleries = ds.gallery
        else:  # 'last'
            # 마지막 데이터셋으로 평가
            _, last_ds, _, _ = loaded[-1]
            queries = last_ds.query
            galleries = last_ds.gallery

        # 최종 할당
        self.train = trains
        self.query = queries
        self.gallery = galleries

        # 통계 계산
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = \
            self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = \
            self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = \
            self.get_imagedata_info(self.gallery)

        if verbose:
            print("\n=> CombinedReID loaded ({} datasets)".format(len(self.specs)))
            self.print_dataset_statistics(self.train, self.query, self.gallery)
