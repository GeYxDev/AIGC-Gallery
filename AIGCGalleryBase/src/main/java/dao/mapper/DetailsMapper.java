package dao.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import dao.entity.Details;
import org.apache.ibatis.annotations.Param;

import java.util.List;

public interface DetailsMapper extends BaseMapper<Details> {
    List<Details> selectWorksDetail(@Param("groupId") Integer groupId);
}
