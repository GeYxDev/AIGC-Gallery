package dao.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import dao.entity.Works;
import dao.mapper.*;
import dao.service.ImagesService;
import dao.service.VideosService;
import dao.service.WorksService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import util.Feedback;

import java.util.List;

@Service
public class WorksServiceImpl extends ServiceImpl<WorksMapper, Works> implements WorksService {

    private final ImagesService imagesService;
    private final VideosService videosService;

    @Autowired
    public WorksServiceImpl(ImagesService imagesService, VideosService videosService) {
        this.imagesService = imagesService;
        this.videosService = videosService;
    }

    @Override
    public Feedback submitWork(Works works) {
        try {
            // 身份验证由上级完成
            boolean saveResult = save(works);
            if (saveResult) {
                return new Feedback(true, "作品元数据上传成功");
            } else {
                return new Feedback(false, "作品元数据上传失败，数据库插入异常");
            }
        } catch (Exception e) {
            return new Feedback(false, "作品元数据上传失败" + e.getMessage());
        }
    }

    @Override
    public Feedback getWorkByGroupId(int groupId) {
        try {
            List<Works> storedWorks =
                    getBaseMapper().selectList(new LambdaQueryWrapper<Works>().eq(Works::getGroupId, groupId));
            if (storedWorks == null) {
                return new Feedback(false, "作品元数据不存在");
            } else {
                return new Feedback(true, storedWorks);
            }
        } catch (Exception e) {
            return new Feedback(false, "作品元数据拉取失败" + e.getMessage());
        }
    }

    @Override
    public Feedback getWorkByWorkId(int workId) {
        try {
            Works storedWork = getOne(new LambdaQueryWrapper<Works>().eq(Works::getWorkId, workId));
            if (storedWork == null) {
                return new Feedback(false, "作品元数据不存在");
            } else {
                return new Feedback(true, storedWork);
            }
        } catch (Exception e) {
            return new Feedback(false, "作品元数据拉取失败" + e.getMessage());
        }
    }

    @Override
    public Feedback deleteWorkByGroupId(int groupId) {
        try {
            // 身份验证由上级完成
            LambdaQueryWrapper<Works> lambdaQueryWrapper = new LambdaQueryWrapper<>();
            lambdaQueryWrapper.eq(Works::getGroupId, groupId);
            List<Works> workList = list(lambdaQueryWrapper);
            for (Works work : workList) {
                if (work.getType().equals("image")) {
                    Feedback isDeletedImage = imagesService.deleteImage(work.getWorkId());
                    if (!isDeletedImage.isSuccess()) {
                        isDeletedImage.setResult("照片删除失败");
                        return isDeletedImage;
                    }
                } else if (work.getType().equals("video")) {
                    Feedback isDeletedVideo = videosService.deleteVideo(work.getWorkId());
                    if (!isDeletedVideo.isSuccess()) {
                        isDeletedVideo.setResult("视频删除失败");
                        return isDeletedVideo;
                    }
                }
            }
            boolean removeResult = remove(lambdaQueryWrapper);
            if (removeResult) {
                return new Feedback(true, "作品元数据删除成功");
            } else {
                return new Feedback(false, "作品元数据删除失败，数据库删除错误");
            }
        } catch (Exception e) {
            return new Feedback(false, "作品元数据删除失败" + e.getMessage());
        }
    }

    @Override
    public Feedback deleteWorkByWorkId(int workId) {
        try {
            // 身份验证由上级完成
            LambdaQueryWrapper<Works> lambdaQueryWrapper = new LambdaQueryWrapper<>();
            lambdaQueryWrapper.eq(Works::getWorkId, workId);
            boolean removeResult = remove(lambdaQueryWrapper);
            if (removeResult) {
                return new Feedback(true, "作品元数据删除成功");
            } else {
                return new Feedback(false, "作品元数据删除失败，数据库删除错误");
            }
        } catch (Exception e) {
            return new Feedback(false, "作品元数据删除失败" + e.getMessage());
        }
    }
}
