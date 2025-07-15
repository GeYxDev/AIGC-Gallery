package dao.controller;

import com.alibaba.fastjson2.JSON;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import dao.entity.Cards;
import dao.service.CardsService;
import lombok.Getter;
import lombok.Setter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import util.Feedback;
import util.FeedbackException;

import java.util.List;

@RestController
@RequestMapping("/roam")
public class CardsController {

    private final CardsService cardsService;

    @Autowired
    public CardsController(CardsService cardsService) {
        this.cardsService = cardsService;
    }

    @Setter
    @Getter
    public static class CardResponse {
        private List<Cards> cardList;
        private boolean hasNextPage;

        public CardResponse(List<Cards> cardList, boolean hasNextPage) {
            this.cardList = cardList;
            this.hasNextPage = hasNextPage;
        }
    }

    @Setter
    @Getter
    public static class SearchResponse {
        private List<Cards> searchResultList;
        private boolean hasNextPage;

        public SearchResponse(List<Cards> searchResultList, boolean hasNextPage) {
            this.searchResultList = searchResultList;
            this.hasNextPage = hasNextPage;
        }
    }

    @PostMapping("/getCards")
    public String getCards(@RequestParam("page") int currentPage, @RequestParam("size") int pageSize) {
        Page<Cards> page = cardsService.getGardList(currentPage, pageSize);
        CardResponse cardResponse = new CardResponse(page.getRecords(), page.hasNext());
        return JSON.toJSONString(cardResponse);
    }

    @PostMapping("/getSearch")
    public String getSearch(@RequestParam("keywords") String keywords,
                            @RequestParam("page") int currentPage, @RequestParam("size") int pageSize) {
        try {
            Page<Cards> page = cardsService.getSearchList(keywords, currentPage, pageSize);
            SearchResponse searchResponse = new SearchResponse(page.getRecords(), page.hasNext());
            return JSON.toJSONString(new Feedback(true, searchResponse));
        } catch (FeedbackException e) {
            return JSON.toJSONString(new Feedback(e.isSuccess(), e.getResult()));
        }
    }
}
