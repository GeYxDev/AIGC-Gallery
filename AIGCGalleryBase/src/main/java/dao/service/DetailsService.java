package dao.service;

import com.baomidou.mybatisplus.extension.service.IService;
import dao.entity.Details;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
public interface DetailsService extends IService<Details> {
    @Transactional
    List<Details> getWorksDetail(int groupId);
}
