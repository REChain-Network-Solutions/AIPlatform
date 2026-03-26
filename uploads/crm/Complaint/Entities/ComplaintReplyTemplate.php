<?php

namespace Modules\Complaint\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;

/**
 * App\Models\ComplaintReplyTemplate
 *
 * @property int $id
 * @property string $reply_heading
 * @property string $reply_text
 * @property \Illuminate\Support\Carbon|null $created_at
 * @property \Illuminate\Support\Carbon|null $updated_at
 * @property-read mixed $icon
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintReplyTemplate newModelQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintReplyTemplate newQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintReplyTemplate query()
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintReplyTemplate whereCreatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintReplyTemplate whereId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintReplyTemplate whereReplyHeading($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintReplyTemplate whereReplyText($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintReplyTemplate whereUpdatedAt($value)
 * @mixin \Eloquent
 * @property int|null $company_id
 * @property-read \App\Models\Company|null $company
 * @method static \Illuminate\Database\Eloquent\Builder|ComplaintReplyTemplate whereCompanyId($value)
 */
class ComplaintReplyTemplate extends BaseModel
{

    use HasCompany;

    protected $table = 'complaint_reply_templates';

    protected $guarded = ['id'];

    //
}
