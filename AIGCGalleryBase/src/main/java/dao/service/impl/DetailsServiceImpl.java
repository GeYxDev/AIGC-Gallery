package dao.service.impl;

import dao.entity.Details;
import dao.mapper.DetailsMapper;
import dao.service.DetailsService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class DetailsServiceImpl extends ServiceImpl<DetailsMapper, Details> implements DetailsService {

    @Override
    public List<Details> getWorksDetail(int groupId) {
        return getBaseMapper().selectWorksDetail(groupId);
    }
}
