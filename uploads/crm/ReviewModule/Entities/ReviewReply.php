<?php

namespace Modules\ReviewModule\Entities;

use App\Traits\HasUuid;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Modules\ReviewModule\Traits\CompanyScoped;

class ReviewReply extends Model
{
    use CompanyScoped;
    use HasFactory;
    use HasUuid;

    protected $fillable = [];

    protected static function newFactory()
    {
        return \Modules\ReviewModule\Database\factories\ReviewReplyFactory::new();
    }
}