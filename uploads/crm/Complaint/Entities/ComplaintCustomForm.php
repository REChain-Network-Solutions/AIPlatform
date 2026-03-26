<?php

namespace Modules\Complaint\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;
use App\Models\CustomField;
use App\Traits\CustomFieldsTrait;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/**
 * App\Models\ComplaintCustomForm
 *
 * @property int $id
 * @property string $field_display_name
 * @property string $field_name
 * @property string $field_type
 * @property int $field_order
 * @property string $status
 * @property \Illuminate\Support\Carbon|null $created_at
 * @property \Illuminate\Support\Carbon|null $updated_at
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintCustomForm newModelQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintCustomForm newQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintCustomForm query()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintCustomForm whereCreatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintCustomForm whereFieldDisplayName($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintCustomForm whereFieldName($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintCustomForm whereFieldOrder($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintCustomForm whereFieldType($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintCustomForm whereId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintCustomForm whereStatus($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintCustomForm whereUpdatedAt($value)
 * @mixin \Eloquent
 * @property int $required
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintCustomForm whereRequired($value)
 * @property int|null $company_id
 * @property int|null $custom_fields_id
 * @property-read \App\Models\Company|null $company
 * @property-read \App\Models\CustomField|null $customField
 * @property-read mixed $extras
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintCustomForm whereCompanyId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintCustomForm whereCustomFieldsId($value)
 */
class ComplaintCustomForm extends BaseModel
{

    use CustomFieldsTrait;
    use HasCompany;

    protected $guarded = ['id'];

    public function customField(): BelongsTo
    {
        return $this->belongsTo(CustomField::class, 'custom_fields_id');
    }

}
