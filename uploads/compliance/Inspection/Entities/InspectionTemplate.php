<?php

namespace Modules\Inspection\Entities;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;

class InspectionTemplate extends Model
{
    use HasFactory;

    protected $table = 'inspection_templates';

    protected $fillable = [
        'name',
        'trade',
        'description',
        'is_active',
    ];

    protected $casts = [
        'is_active' => 'boolean',
    ];

    public function items()
    {
        return $this->hasMany(InspectionTemplateItem::class, 'template_id')->orderBy('sort_order');
    }
}
