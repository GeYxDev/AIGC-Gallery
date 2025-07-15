package dao.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.houbb.sensitive.word.core.SensitiveWordHelper;
import dao.entity.*;
import dao.mapper.*;
import dao.service.*;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Service;
import util.Feedback;
import util.FeedbackException;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.ZoneId;
import java.util.*;

@Service
public class GroupsServiceImpl extends ServiceImpl<GroupsMapper, Groups> implements GroupsService {

    private final GroupsMapper groupsMapper;
    private final UsersMapper usersMapper;
    private final FollowsMapper followsMapper;
    private final WorksMapper worksMapper;
    private final ImagesMapper imagesMapper;
    private final VideosMapper videosMapper;
    private final CommentsMapper commentsMapper;
    private final WorksService worksService;
    private final ImagesService imagesService;
    private final VideosService videosService;
    private final CommentsService commentsService;
    private final ResourceLoader resourceLoader;

    @Autowired
    public GroupsServiceImpl(GroupsMapper groupsMapper, UsersMapper usersMapper, FollowsMapper followsMapper,
                             WorksMapper worksMapper, ImagesMapper imagesMapper, CommentsMapper commentsMapper,
                             ResourceLoader resourceLoader, VideosMapper videosMapper, CommentsService commentsService,
                             ImagesService imagesService, VideosService videosService, WorksService worksService) {
        this.groupsMapper = groupsMapper;
        this.usersMapper = usersMapper;
        this.followsMapper = followsMapper;
        this.worksMapper = worksMapper;
        this.imagesMapper = imagesMapper;
        this.videosMapper = videosMapper;
        this.commentsMapper = commentsMapper;
        this.worksService = worksService;
        this.imagesService = imagesService;
        this.videosService = videosService;
        this.commentsService = commentsService;
        this.resourceLoader = resourceLoader;
    }

    @Override
    public Map<String, Object> getMomentByAccount(String account, int currentPage, int pageSize) {
        // 获取关注列表
        List<String> followedAccountList = followsMapper.selectList(new LambdaQueryWrapper<Follows>()
                .eq(Follows::getFollowerAccount, account)).stream().map(Follows::getFollowedAccount).toList();
        // 获得完整账号列表
        List<String> allAccountList = new ArrayList<>();
        allAccountList.add(account);
        allAccountList.addAll(followedAccountList);
        // 分页查询所有动态
        IPage<Groups> page = new Page<>(currentPage, pageSize);
        IPage<Groups> groupPage = groupsMapper.selectPage(page, new LambdaQueryWrapper<Groups>()
                .in(Groups::getAccount, allAccountList).orderByDesc(Groups::getGroupId));
        List<Map<String, Object>> momentList = new ArrayList<>();
        for (Groups group : groupPage.getRecords()) {
            Map<String, Object> moment = new HashMap<>();
            moment.put("groupId", group.getGroupId());
            moment.put("account", group.getAccount());
            moment.put("theme", group.getTheme());
            moment.put("text", group.getContent());
            if (group.getCreateTime() != null) {
                Date createTime = Date.from(group.getCreateTime().atZone(ZoneId.systemDefault()).toInstant());
                moment.put("createTime", createTime);
            } else {
                moment.put("createTime", null);
            }
            // 获得用户信息
            Users user = usersMapper.selectById(group.getAccount());
            moment.put("nickname", user.getNickname());
            moment.put("avatar", user.getAvatar());
            // 获取作品信息
            Works work = worksMapper.selectOne(new QueryWrapper<Works>().select("TOP 1 *")
                    .eq("group_id", group.getGroupId()).orderByDesc("work_id"));
            if (work != null) {
                moment.put("type", work.getType());
                moment.put("aspectRatio", work.getAspectRatio());
                if ("video".equals(work.getType())) {
                    Videos video = videosMapper.selectById(work.getWorkId());
                    moment.put("media", Map.of(
                            "workId", work.getWorkId(),
                            "cover", video.getCover(),
                            "videoLink", video.getVideoLink(),
                            "mediaRatio", work.getAspectRatio()
                    ));
                    moment.put("workNum", 1);
                } else if ("image".equals(work.getType())) {
                    List<Integer> workList = worksMapper.selectObjs(new LambdaQueryWrapper<Works>()
                            .eq(Works::getGroupId, group.getGroupId()).select(Works::getWorkId));
                    List<Images> imageList = imagesMapper.selectList(
                            new LambdaQueryWrapper<Images>().in(Images::getWorkId, workList));
                    List<Map<String, Object>> mediaList = new ArrayList<>();
                    for (Images image : imageList) {
                        mediaList.add(Map.of("workId", image.getWorkId(), "image", image.getImage()));
                    }
                    moment.put("media", mediaList);
                    moment.put("workNum", workList.size());
                }
            }
            // 获取评论信息
            List<Comments> commentList = commentsMapper.selectList(
                    new LambdaQueryWrapper<Comments>().eq(Comments::getGroupId, group.getGroupId()));
            List<Map<String, Object>> commentItemList = new ArrayList<>();
            for (Comments comment : commentList) {
                Users commentUser = usersMapper.selectById(comment.getAccount());
                commentItemList.add(Map.of(
                        "commentId", comment.getCommentId(),
                        "nickname", commentUser.getNickname(),
                        "content", comment.getContent()
                ));
            }
            moment.put("comment", commentItemList);
            momentList.add(moment);
        }
        // 封装分页信息
        Map<String, Object> getMomentResult = new HashMap<>();
        boolean hasNext = groupPage.getCurrent() < groupPage.getPages();
        getMomentResult.put("hasNextPage", hasNext);
        getMomentResult.put("momentList", momentList);
        return getMomentResult;
    }

    @Override
    public Map<String, Object> getIndividualMoment(String account, int currentPage, int pageSize) {
        // 分页查询所有个人动态
        IPage<Groups> page = new Page<>(currentPage, pageSize);
        IPage<Groups> groupPage = groupsMapper.selectPage(page, new LambdaQueryWrapper<Groups>()
                .eq(Groups::getAccount, account).orderByDesc(Groups::getGroupId));
        List<Map<String, Object>> momentList = new ArrayList<>();
        for (Groups group : groupPage.getRecords()) {
            Map<String, Object> moment = new HashMap<>();
            moment.put("groupId", group.getGroupId());
            moment.put("account", group.getAccount());
            moment.put("theme", group.getTheme());
            moment.put("text", group.getContent());
            if (group.getCreateTime() != null) {
                Date createTime = Date.from(group.getCreateTime().atZone(ZoneId.systemDefault()).toInstant());
                moment.put("createTime", createTime);
            } else {
                moment.put("createTime", null);
            }
            // 获取作品信息
            Works work = worksMapper.selectOne(new QueryWrapper<Works>().select("TOP 1 *")
                    .eq("group_id", group.getGroupId()).orderByDesc("work_id"));
            if (work != null) {
                moment.put("type", work.getType());
                moment.put("aspectRatio", work.getAspectRatio());
                if ("video".equals(work.getType())) {
                    Videos video = videosMapper.selectById(work.getWorkId());
                    moment.put("media", Map.of(
                            "workId", work.getWorkId(),
                            "cover", video.getCover(),
                            "videoLink", video.getVideoLink(),
                            "mediaRatio", work.getAspectRatio()
                    ));
                    moment.put("workNum", 1);
                } else if ("image".equals(work.getType())) {
                    List<Integer> workList = worksMapper.selectObjs(new LambdaQueryWrapper<Works>()
                            .eq(Works::getGroupId, group.getGroupId()).select(Works::getWorkId));
                    List<Images> imageList = imagesMapper.selectList(
                            new LambdaQueryWrapper<Images>().in(Images::getWorkId, workList));
                    List<Map<String, Object>> mediaList = new ArrayList<>();
                    for (Images image : imageList) {
                        mediaList.add(Map.of("workId", image.getWorkId(), "image", image.getImage()));
                    }
                    moment.put("media", mediaList);
                    moment.put("workNum", workList.size());
                }
            }
            momentList.add(moment);
        }
        // 按时间进行分组
        List<Map<String, Object>> groupedMomentList = new ArrayList<>();
        for (Map<String, Object> moment : momentList) {
            Date createTime = (Date) moment.get("createTime");
            if (createTime != null) {
                Calendar calendar = Calendar.getInstance();
                calendar.setTime(createTime);
                String year = String.valueOf(calendar.get(Calendar.YEAR));
                String month = String.format("%02d", calendar.get(Calendar.MONTH) + 1);
                String day = String.format("%02d", calendar.get(Calendar.DAY_OF_MONTH));
                Map<String, Object> currentYearMap = null;
                for (Map<String, Object> yearMap : groupedMomentList) {
                    if (yearMap.containsKey("year") && yearMap.get("year").equals(year)) {
                        currentYearMap = yearMap;
                        break;
                    }
                }
                if (currentYearMap == null) {
                    currentYearMap = new LinkedHashMap<>();
                    currentYearMap.put("year", year);
                    currentYearMap.put("content", new ArrayList<>());
                    groupedMomentList.add(currentYearMap);
                }
                @SuppressWarnings("unchecked")
                List<Map<String, Object>> yearList = (List<Map<String, Object>>) currentYearMap.get("content");
                Map<String, Object> currentMonthDayMap = null;
                for (Map<String, Object> monthDayMap : yearList) {
                    if (monthDayMap.containsKey("day") && monthDayMap.containsKey("month") &&
                            monthDayMap.get("day").equals(day) && monthDayMap.get("month").equals(month)) {
                        currentMonthDayMap = monthDayMap;
                        break;
                    }
                }
                if (currentMonthDayMap == null) {
                    currentMonthDayMap = new LinkedHashMap<>();
                    currentMonthDayMap.put("tag", year + month + day + moment.get("groupId").toString());
                    currentMonthDayMap.put("month", month);
                    currentMonthDayMap.put("day", day);
                    currentMonthDayMap.put("content", new ArrayList<>());
                    yearList.add(currentMonthDayMap);
                }
                @SuppressWarnings("unchecked")
                List<Map<String, Object>> contentList = (List<Map<String, Object>>) currentMonthDayMap.get("content");
                contentList.add(moment);
            }
        }
        // 封装分页信息
        Map<String, Object> getMomentResult = new HashMap<>();
        boolean hasNext = groupPage.getCurrent() < groupPage.getPages();
        getMomentResult.put("hasNextPage", hasNext);
        getMomentResult.put("momentList", groupedMomentList);
        return getMomentResult;
    }

    @Override
    public Feedback addGroupLikes(int groupId) {
        try {
            Groups group = groupsMapper.selectById(groupId);
            if (group == null) {
                return new Feedback(false, "作品组不存在");
            }
            LambdaUpdateWrapper<Groups> updateWrapper = new LambdaUpdateWrapper<>();
            updateWrapper.eq(Groups::getGroupId, groupId);
            updateWrapper.setSql("likes = likes + 1");
            int updateResult = groupsMapper.update(null, updateWrapper);
            if (updateResult > 0) {
                return new Feedback(true, "作品组点赞成功");
            } else {
                return new Feedback(false, "作品组点赞失败");
            }
        } catch (Exception e) {
            return new Feedback(false, "作品组点赞失败" + e.getMessage());
        }
    }

    @Override
    public Feedback submitGroup(Groups groups) {
        try {
            // 身份验证由上级完成
            boolean saveResult = save(groups);
            if (saveResult) {
                return new Feedback(true, "作品组上传成功");
            } else {
                return new Feedback(false, "作品组上传失败，数据库插入异常");
            }
        } catch (Exception e) {
            return new Feedback(false, "作品组上传失败" + e.getMessage());
        }
    }

    @Override
    public Feedback getGroupByGroupId(int groupId) {
        try {
            Groups storedGroup = getOne(new LambdaQueryWrapper<Groups>().eq(Groups::getGroupId, groupId));
            if (storedGroup == null) {
                return new Feedback(false, "作品组不存在");
            } else {
                return new Feedback(true, storedGroup);
            }
        } catch (Exception e) {
            return new Feedback(false, "作品组拉取失败" + e.getMessage());
        }
    }

    @Override
    public Feedback getGroupByAccount(String account) {
        try {
            Groups storedGroup = getOne(new LambdaQueryWrapper<Groups>().eq(Groups::getAccount, account));
            if (storedGroup == null) {
                return new Feedback(false, "作品组不存在");
            } else {
                return new Feedback(true, storedGroup);
            }
        } catch (Exception e) {
            return new Feedback(false, "作品组拉取失败" + e.getMessage());
        }
    }

    @Override
    public Feedback deleteGroupByGroupId(String account, int groupId) {
        try {
            // 身份验证由上级完成
            LambdaQueryWrapper<Groups> lambdaQueryWrapper = new LambdaQueryWrapper<>();
            lambdaQueryWrapper.eq(Groups::getGroupId, groupId);
            Groups group = getOne(lambdaQueryWrapper);
            if (group == null) {
                return new Feedback(false, "作品组不存在");
            }
            if (!group.getAccount().equals(account)) {
                return new Feedback(false, "无删除权限");
            }
            Feedback isDeletedWork = worksService.deleteWorkByGroupId(group.getGroupId());
            if (!isDeletedWork.isSuccess()) {
                isDeletedWork.setResult("作品元数据或作品删除失败");
                return isDeletedWork;
            }
            Feedback isDeletedComment = commentsService.deleteCommentByGroupId(group.getGroupId());
            if (!isDeletedComment.isSuccess()) {
                isDeletedComment.setResult("评论删除失败");
                return isDeletedComment;
            }
            boolean removeResult = remove(lambdaQueryWrapper);
            if (removeResult) {
                return new Feedback(true, "作品组删除成功");
            } else {
                return new Feedback(false, "作品组删除失败，数据库删除错误");
            }
        } catch (Exception e) {
            return new Feedback(false, "作品组删除失败" + e.getMessage());
        }
    }

    @Override
    public Feedback deleteGroupByAccount(String account) {
        try {
            // 身份验证由上级完成
            LambdaQueryWrapper<Groups> lambdaQueryWrapper = new LambdaQueryWrapper<>();
            lambdaQueryWrapper.eq(Groups::getAccount, account);
            boolean removeResult = remove(lambdaQueryWrapper);
            if (removeResult) {
                return new Feedback(true, "作品组删除成功");
            } else {
                return new Feedback(false, "作品组删除失败，数据库删除错误");
            }
        } catch (Exception e) {
            return new Feedback(false, "作品组删除失败" + e.getMessage());
        }
    }

    @Override
    public Feedback submitMoment(String account, String theme, String content, String artList, String artType) {
        try {
            // 解析作品列表
            List<Map<String, Object>> artListMap = new ObjectMapper().readValue(artList, new TypeReference<>() {});
            if (artListMap == null || artListMap.isEmpty()) {
                throw new FeedbackException(false, "作品列表为空");
            }
            // 插入动态数据至数据库
            Groups groups = new Groups();
            groups.setAccount(account);
            groups.setTheme(SensitiveWordHelper.replace(theme));
            groups.setContent(SensitiveWordHelper.replace(content));
            boolean saveGroupResult = save(groups);
            if (!saveGroupResult) {
                throw new FeedbackException(false, "作品组上传失败");
            }
            // 获取动态数据唯一标识符
            Integer groupId = groups.getGroupId();
            if (groupId == null) {
                throw new FeedbackException(false, "获取动态唯一标识符失败");
            }
            for (Map<String, Object> artMap : artListMap) {
                // 插入作品元数据至数据库
                Works works = new Works();
                works.setGroupId(groupId);
                works.setAspectRatio(Float.parseFloat(artMap.get("aspectRatio").toString()));
                works.setType((String) artMap.get("type"));
                works.setModelType((String) artMap.get("modelType"));
                works.setMediaWidth((Integer) artMap.get("mediaWidth"));
                works.setMediaHeight((Integer) artMap.get("mediaHeight"));
                works.setIteration((Integer) artMap.get("iteration"));
                Feedback saveWorkResult = worksService.submitWork(works);
                if (!saveWorkResult.isSuccess()) {
                    throw new FeedbackException(false, (String) saveWorkResult.getResult());
                }
                // 获取作品元数据唯一标识符
                Integer workId = works.getWorkId();
                if (workId == null) {
                    throw new FeedbackException(false, "获取作品元数据唯一标识符失败");
                }
                if (artType.equals("image")) {
                    // 若作品为图像
                    Images images = new Images();
                    images.setWorkId(workId);
                    images.setImage((String) artMap.get("image"));
                    Feedback saveImageResult = imagesService.submitImage(images);
                    if (!saveImageResult.isSuccess()) {
                        throw new FeedbackException(false, (String) saveImageResult.getResult());
                    }
                } else if (artType.equals("video")) {
                    // 若作品为视频
                    Videos videos = new Videos();
                    videos.setWorkId(workId);
                    videos.setCover((String) artMap.get("cover"));
                    // 保存视频至本地并生成访问链接
                    String base64Video = ((String) artMap.get("video"))
                            .replaceAll("^data:video/[\\w+-]+;base64,", "");
                    byte[] videoBytes = Base64.getDecoder().decode(base64Video);
                    String videoFileName = UUID.randomUUID() + ".mp4";
                    Resource resource = resourceLoader.getResource("file:src/main/video/");
                    String videoDir = resource.getFile().getAbsolutePath();
                    Files.write(Path.of(videoDir + '\\' + videoFileName), videoBytes);
                    String videoLink = "http://127.0.0.1:8081/video/" + videoFileName;
                    videos.setVideoLink(videoLink);
                    Feedback saveVideoResult = videosService.submitVideo(videos);
                    if (!saveVideoResult.isSuccess()) {
                        throw new FeedbackException(false, (String) saveVideoResult.getResult());
                    }
                } else {
                    // 未知作品类型
                    throw new FeedbackException(false, "作品类型错误");
                }
            }
            return new Feedback(true, "动态上传成功");
        } catch (FeedbackException e) {
            throw e;
        } catch (Exception e) {
            throw new FeedbackException(false, "动态上传失败" + e.getMessage());
        }
    }
}
