package dao.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import dao.entity.Images;
import dao.mapper.ImagesMapper;
import dao.service.ImagesService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.stereotype.Service;
import util.Feedback;

@Service
public class ImagesServiceImpl extends ServiceImpl<ImagesMapper, Images> implements ImagesService {

    @Override
    public Feedback submitImage(Images images) {
        try {
            // 身份验证由上级完成
            boolean saveResult = save(images);
            if (saveResult) {
                return new Feedback(true, "照片上传成功");
            } else {
                return new Feedback(false, "照片上传失败，数据库插入异常");
            }
        } catch (Exception e) {
            return new Feedback(false, "照片上传失败" + e.getMessage());
        }
    }

    @Override
    public Feedback getImage(int workId) {
        try {
            Images storedImage = getOne(new LambdaQueryWrapper<Images>().eq(Images::getWorkId, workId));
            if (storedImage == null) {
                return new Feedback(false, "照片不存在");
            } else {
                return new Feedback(true, storedImage);
            }
        } catch (Exception e) {
            return new Feedback(false, "照片拉取失败" + e.getMessage());
        }
    }

    @Override
    public Feedback deleteImage(int workId) {
        try {
            // 身份验证由上级完成
            LambdaQueryWrapper<Images> lambdaQueryWrapper = new LambdaQueryWrapper<>();
            lambdaQueryWrapper.eq(Images::getWorkId, workId);
            boolean removeResult = remove(lambdaQueryWrapper);
            if (removeResult) {
                return new Feedback(true, "照片删除成功");
            } else {
                return new Feedback(false, "照片删除失败，数据库删除错误");
            }
        } catch (Exception e) {
            return new Feedback(false, "照片删除失败" + e.getMessage());
        }
    }
}
