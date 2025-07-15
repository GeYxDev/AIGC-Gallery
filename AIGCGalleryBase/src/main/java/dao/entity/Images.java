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
@TableName("Images")
public class Images implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    @TableId(value = "work_id", type = IdType.INPUT)
    private Integer workId;

    private String image;


    @Override
    public String toString() {
        return "Images{" +
        "workId=" + workId +
        ", image=" + image +
        "}";
    }
}
