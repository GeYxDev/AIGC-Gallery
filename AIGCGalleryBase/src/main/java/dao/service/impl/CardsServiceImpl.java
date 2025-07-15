package dao.service.impl;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.huaban.analysis.jieba.JiebaSegmenter;
import dao.entity.*;
import dao.mapper.CardsMapper;
import dao.service.CardsService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.stereotype.Service;
import util.FeedbackException;

import java.util.List;

@Service
public class CardsServiceImpl extends ServiceImpl<CardsMapper, Cards> implements CardsService {

    @Override
    public Page<Cards> getGardList(int currentPage, int pageSize) {
        Page<Cards> page = new Page<>(currentPage, pageSize);
        // 执行分页查询
        return getBaseMapper().selectCardList(page);
    }

    @Override
    public Page<Cards> getSearchList(String keywords, int currentPage, int pageSize) {
        try {
            JiebaSegmenter js = new JiebaSegmenter();
            List<String> keywordList = js.sentenceProcess(keywords);
            Page<Cards> page = new Page<>(currentPage, pageSize);
            // 执行关键词分页搜索
            return getBaseMapper().selectSearchList(page, keywordList);
        } catch (Exception e) {
            throw new FeedbackException(false, "作品搜索失败" + e.getMessage());
        }
    }
}
