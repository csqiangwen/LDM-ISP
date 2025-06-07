import torch


class ImageSpliterTh:
    def __init__(self, im, pch_size, stride):
        '''
        Input:
            im: n x c x h x w, torch tensor, float, low-resolution image in SR
            pch_size, stride: patch setting
            sf: scale factor in image super-resolution
        '''
        assert stride <= pch_size
        self.stride = stride
        self.pch_size = pch_size

        bs, chn, height, width= im.shape
        chn = 3
        self.height_starts_list = self.extract_starts(height)
        self.width_starts_list = self.extract_starts(width)
        self.num_pchs = 0

        self.im_ori = im
        self.im_res = torch.zeros([bs, chn, height, width], dtype=im.dtype, device=im.device)
        self.pixel_count = torch.zeros([bs, chn, height, width], dtype=im.dtype, device=im.device)

        self.column_merges = {}

    def extract_starts(self, length):
        if length <= self.pch_size:
            starts = [0,]
        else:
            starts = list(range(0, length, self.stride))
            for i in range(len(starts)):
                if starts[i] + self.pch_size > length:
                    starts[i] = length - self.pch_size
            starts = sorted(set(starts), key=starts.index)
        return starts

    def extract_patches(self):
        patches_dict = {}
        for width_start_list in self.width_starts_list:
            w_start = width_start_list
            w_end = w_start + self.pch_size
            for height_start_list in self.height_starts_list:
                h_start = height_start_list
                h_end = h_start + self.pch_size

                pch = self.im_ori[:, :, h_start:h_end, w_start:w_end]
                
                patches_dict['(%d, %d)'%(h_start, w_start)] = [pch, (h_start, h_end, w_start, w_end)]
        
        return patches_dict

    def update(self, pch_res, index_infos):
        '''
        Input:
            pch_res: n x c x pch_size x pch_size, float
            index_infos: (h_start, h_end, w_start, w_end)
        '''

        if index_infos[0] == 0:
            canvas = self.im_res.clone()
            canvas[:, :, index_infos[0]:index_infos[1], index_infos[2]:index_infos[3]] = pch_res
            self.column_merges['(%d, %d)'%(index_infos[2], index_infos[3])] = [canvas, index_infos]
        else:
            old_canvas, old_index_infos = self.column_merges['(%d, %d)'%(index_infos[2], index_infos[3])]
            merge_weights = torch.linspace(1, 0, old_index_infos[1] - index_infos[0]).unsqueeze(0).unsqueeze(0).unsqueeze(3).cuda()
            
            old_canvas[:, :, index_infos[0]:old_index_infos[1], index_infos[2]:old_index_infos[3]] = old_canvas[:, :, index_infos[0]:old_index_infos[1], index_infos[2]:old_index_infos[3]] * merge_weights +\
                                                                                                     pch_res[:, :, 0:old_index_infos[1]-index_infos[0], :] * (1 - merge_weights)

            old_canvas[:, :, old_index_infos[1]:index_infos[1], index_infos[2]:old_index_infos[3]] = pch_res[:, :, old_index_infos[1]-index_infos[0]:, :]
            self.column_merges['(%d, %d)'%(index_infos[2], index_infos[3])] = [old_canvas, index_infos]

    def gather(self):
        keys_list = list(self.column_merges.keys())
        self.im_res, prev_index_infos = self.column_merges[keys_list[0]]
        for key in keys_list[1:]:
            canvas, curr_index_infos = self.column_merges[key]
            merge_weights = torch.linspace(1, 0, prev_index_infos[3] - curr_index_infos[2]).unsqueeze(0).unsqueeze(0).unsqueeze(0).cuda()

            self.im_res[:, :, :, curr_index_infos[2]:prev_index_infos[3]] = self.im_res[:, :, :, curr_index_infos[2]:prev_index_infos[3]] * merge_weights +\
                                                                            canvas[:, :, :, curr_index_infos[2]:prev_index_infos[3]] * (1-merge_weights)
            self.im_res[:, :, :, prev_index_infos[3]:curr_index_infos[3]] = canvas[:, :, :, prev_index_infos[3]:curr_index_infos[3]]
            prev_index_infos = curr_index_infos

        return self.im_res