<?php
namespace Modules\Feedback\Entities;
use Illuminate\Database\Eloquent\Model;

class Feedback extends Model
{
    protected $table = 'feedback_items';
    protected $guarded = [];
}
