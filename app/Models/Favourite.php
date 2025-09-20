<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\HasOne;

class Favourite extends Model
{
    use HasFactory;

    protected $table = 'favourite_list';

    protected $guarded = [];

    /**
     * return favorited open ai generator categories
     */
    public function openaiGeneratorChatCategory(): HasOne
    {
        return $this->hasOne(OpenaiGeneratorChatCategory::class, 'id', 'item_id');
    }
}
