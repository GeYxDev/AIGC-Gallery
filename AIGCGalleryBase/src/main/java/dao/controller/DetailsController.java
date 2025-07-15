package dao.controller;

import com.alibaba.fastjson2.JSON;
import dao.entity.Details;
import dao.entity.Groups;
import dao.service.DetailsService;
import dao.service.GroupsService;
import lombok.Getter;
import lombok.Setter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.time.format.DateTimeFormatter;
import java.util.List;

@RestController
@RequestMapping("/detail")
public class DetailsController {

    private final DetailsService detailsService;
    private final GroupsService groupsService;

    @Autowired
    public DetailsController(DetailsService detailsService, GroupsService groupsService) {
        this.detailsService = detailsService;
        this.groupsService = groupsService;
    }

    @Setter
    @Getter
    public static class DetailResponse {
        private List<Details> mediaDetailList;
        private String text;
        private String createTime;

        public DetailResponse(List<Details> mediaDetailList, String text, String createTime) {
            this.mediaDetailList = mediaDetailList;
            this.text = text;
            this.createTime = createTime;
        }
    }

    @PostMapping("/getArtDetail")
    public String getArtDetail(@RequestParam("groupId") int groupId) {
        List<Details> mediaDetailList = detailsService.getWorksDetail(groupId);
        Groups groups = (Groups) groupsService.getGroupByGroupId(groupId).getResult();
        DateTimeFormatter dateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        DetailResponse detailResponse = new DetailResponse(
                mediaDetailList,
                groups.getContent(),
                groups.getCreateTime().format(dateTimeFormatter)
        );
        return JSON.toJSONString(detailResponse);
    }
}
