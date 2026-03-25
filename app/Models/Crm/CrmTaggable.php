<?php

namespace App\Models\Crm;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\MorphTo;

class CrmTaggable extends Model
{
    protected $table = 'crm_taggables';

    protected $fillable = ['crm_tag_id', 'taggable_id', 'taggable_type'];

    public function tag(): BelongsTo
    {
        return $this->belongsTo(CrmTag::class, 'crm_tag_id');
    }

    public function taggable(): MorphTo
    {
        return $this->morphTo();
    }
}
