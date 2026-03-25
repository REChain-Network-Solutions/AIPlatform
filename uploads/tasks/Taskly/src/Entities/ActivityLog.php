<?php

namespace Workdo\Taskly\Entities;

use App\Models\User;
use Illuminate\Database\Eloquent\Model;

class ActivityLog extends Model
{
    protected $fillable = [
        'user_id','user_type', 'project_id','log_type','remark'
    ];

    public static $user_name;

    public function getRemark(){
        $remark = json_decode($this->remark,true);
        if(is_array($remark)) {

            if($this->user_name != null){
                $user = $this->user;
                $this->user_name = $user ? $user->name : '';
            }

            if ($this->log_type == 'Upload File') {
                return  $this->user_name. ' ' . __('Upload new file') . ' <b>' . $remark['file_name'] . '</b>';
            } elseif ($this->log_type == 'Create Timesheet'){
                return $this->user_name. " " . __('Create new Timesheet');
            } elseif ($this->log_type == 'Create Bug'){
                return $this->user_name . ' ' . __('Create new Bug') . " <b>" . $remark['title'] . "</b>";
            } elseif ($this->log_type == 'Move Bug'){
                return $this->user_name . " " . __('Move Bug') . " <b>" . $remark['title'] . "</b> " . __('from') . " " . __(ucwords($remark['old_status'])) . " " . __('to') . " " . __(ucwords($remark['new_status']));
            } elseif ($this->log_type == 'Invite User'){
                $inviteUser = User::find($remark['user_id']);
                return $this->user_name . ' ' . __('Invite new User') . ' <b>' . (($inviteUser)?$inviteUser->name:'') . '</b>';
            } elseif ($this->log_type == 'Share with Client'){
                $inviteClient = User::find($remark['client_id']);
                return $this->user_name . ' ' . __('Share Project with Client') . ' <b>' . (($inviteClient)?$inviteClient->name:'') . '</b>';
            } elseif ($this->log_type == 'Create Task'){
                return $this->user_name . ' ' . __('Create new Task') . " <b>" . $remark['title'] . "</b>";
            } elseif ($this->log_type == 'Move'){
                return $this->user_name . " " . __('Move Task') . " <b>" . $remark['title'] . "</b> " . __('from') . " " . __(ucwords($remark['old_status'])) . " " . __('to') . " " . __(ucwords($remark['new_status']));
            } elseif ($this->log_type == 'Create Milestone'){
                return $this->user_name . " " . __('Create new Milestone') . " <b>" . $remark['title'] . "</b>";
            } elseif ($this->log_type == 'Share with Vender'){
                $inviteVender = User::find($remark['vender_id']);
                return $this->user_name . " " . __('Share Project with Vender') . " <b>" . (($inviteVender)?$inviteVender->name:'') . "</b>";
            }

        }else{
            return $this->remark;
        }
    }


    public function user(){
        return $this->hasOne('App\Models\User','id','user_id');
    }
}
