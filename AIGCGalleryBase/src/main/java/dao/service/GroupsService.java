package dao.service;

import dao.entity.Groups;
import com.baomidou.mybatisplus.extension.service.IService;
import org.springframework.transaction.annotation.Transactional;
import util.Feedback;

import java.util.Map;

public interface GroupsService extends IService<Groups> {
    @Transactional
    Map<String, Object> getMomentByAccount(String account, int currentPage, int pageSize);
    @Transactional
    Map<String, Object> getIndividualMoment(String account, int currentPage, int pageSize);
    @Transactional
    Feedback addGroupLikes(int groupId);
    @Transactional
    Feedback submitGroup(Groups groups);
    @Transactional
    Feedback getGroupByGroupId(int groupId);
    @Transactional
    Feedback getGroupByAccount(String account);
    @Transactional
    Feedback deleteGroupByGroupId(String account, int groupId);
    @Transactional
    Feedback deleteGroupByAccount(String account);
    @Transactional(rollbackFor = Exception.class)
    Feedback submitMoment(String account, String theme, String content, String artList, String artType);
}
