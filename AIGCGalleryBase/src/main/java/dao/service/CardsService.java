package dao.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import dao.entity.Cards;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public interface CardsService extends IService<Cards> {
    @Transactional
    Page<Cards> getGardList(int currentPage, int pageSize);
    @Transactional
    Page<Cards> getSearchList(String keywords, int currentPage, int pageSize);
}
