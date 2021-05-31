from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # parser.add_argument('--dataroot', type=str, default='/home/Data/AllDataImages/2018_FaceFH', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--dataroot', type=str, default='', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--phase', type=str, default='', help='train, val, test, etc')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.set_defaults(model='test')
        
        self.isTrain = False
        return parser
