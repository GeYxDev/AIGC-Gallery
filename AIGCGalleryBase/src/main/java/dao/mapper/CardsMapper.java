package dao.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import dao.entity.Cards;
import org.apache.ibatis.annotations.Param;

import java.util.List;

public interface CardsMapper extends BaseMapper<Cards> {
    Page<Cards> selectCardList(Page<Cards> page);
    Page<Cards> selectSearchList(Page<Cards> page, @Param("keywords") List<String> keywords);
}
