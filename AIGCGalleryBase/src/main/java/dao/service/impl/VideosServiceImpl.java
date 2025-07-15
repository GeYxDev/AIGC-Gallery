package dao.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import dao.entity.Videos;
import dao.mapper.VideosMapper;
import dao.service.VideosService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.stereotype.Service;
import util.Feedback;

@Service
public class VideosServiceImpl extends ServiceImpl<VideosMapper, Videos> implements VideosService {

    @Override
    public Feedback submitVideo(Videos videos) {
        try {
            // 身份验证由上级完成
            boolean saveResult = save(videos);
            if (saveResult) {
                return new Feedback(true, "视频上传成功");
            } else {
                return new Feedback(false, "视频上传失败，数据库插入异常");
            }
        } catch (Exception e) {
            return new Feedback(false, "视频上传失败" + e.getMessage());
        }
    }

    @Override
    public Feedback getVideo(int workId) {
        try {
            Videos storedVideo = getOne(new LambdaQueryWrapper<Videos>().eq(Videos::getWorkId, workId));
            if (storedVideo == null) {
                return new Feedback(false, "视频不存在");
            } else {
                return new Feedback(true, storedVideo);
            }
        } catch (Exception e) {
            return new Feedback(false, "视频拉取失败" + e.getMessage());
        }
    }

    @Override
    public Feedback deleteVideo(int workId) {
        try {
            // 身份验证由上级完成
            LambdaQueryWrapper<Videos> lambdaQueryWrapper = new LambdaQueryWrapper<>();
            lambdaQueryWrapper.eq(Videos::getWorkId, workId);
            boolean removeResult = remove(lambdaQueryWrapper);
            if (removeResult) {
                return new Feedback(true, "视频删除成功");
            } else {
                return new Feedback(false, "视频删除失败，数据库删除错误");
            }
        } catch (Exception e) {
            return new Feedback(false, "视频删除失败" + e.getMessage());
        }
    }
}
