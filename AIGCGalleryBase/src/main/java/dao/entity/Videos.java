package dao.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Getter;
import lombok.Setter;

import java.io.Serial;
import java.io.Serializable;

@Setter
@Getter
@TableName("Videos")
public class Videos implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    @TableId(value = "work_id", type = IdType.INPUT)
    private Integer workId;

    private String cover;

    private String videoLink;


    @Override
    public String toString() {
        return "Videos{" +
        "workId=" + workId +
        ", cover=" + cover +
        ", videoLink=" + videoLink +
        "}";
    }
}
