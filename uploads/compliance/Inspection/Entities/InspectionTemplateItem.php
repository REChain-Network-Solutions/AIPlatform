<?php

namespace Modules\Inspection\Entities;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;

class InspectionTemplateItem extends Model
{
    use HasFactory;

    protected $table = 'inspection_template_items';

    protected $fillable = [
        'template_id',
        'item_name',
        'standard',
        'sort_order',
        'is_required',
    ];

    protected $casts = [
        'is_required' => 'boolean',
    ];

    public function template()
    {
        return $this->belongsTo(InspectionTemplate::class, 'template_id');
    }
}
