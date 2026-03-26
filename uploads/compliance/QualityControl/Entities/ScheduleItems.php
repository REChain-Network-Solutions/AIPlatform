<?php

namespace Modules\QualityControl\Entities;

use App\Models\BaseModel;
use Illuminate\Database\Eloquent\Relations\HasOne;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/**
 * App\Models\ScheduleItems
 *
 * @property int $id
 * @property int $schedule_id
 * @property int|null $quickbooks_item_id
 * @property string $item_name
 * @property string|null $item_summary
 * @property string $type
 * @property float $quantity
 * @property float $unit_price
 * @property float $amount
 * @property string|null $taxes
 * @property \Illuminate\Support\Carbon|null $created_at
 * @property \Illuminate\Support\Carbon|null $updated_at
 * @property string|null $hsn_sac_code
 * @property-read mixed $icon
 * @method static \Illuminate\Database\Eloquent\Builder|ScheduleItems newModelQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ScheduleItems newQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ScheduleItems query()
 * @method static \Illuminate\Database\Eloquent\Builder|ScheduleItems whereAmount($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ScheduleItems whereCreatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ScheduleItems whereHsnSacCode($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ScheduleItems whereId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ScheduleItems whereScheduleId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ScheduleItems whereItemName($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ScheduleItems whereItemSummary($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ScheduleItems whereQuantity($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ScheduleItems whereTaxes($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ScheduleItems whereType($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ScheduleItems whereUnitPrice($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ScheduleItems whereUpdatedAt($value)
 * @mixin \Eloquent
 * @property-read \App\Models\ScheduleItemImage|null $scheduleItemImage
 * @property-read mixed $tax_list
 */
class ScheduleItems extends BaseModel
{

    /**
     * Inspection module table (prefixed to avoid collisions with other modules / core).
     */
    protected $table = 'inspection_schedule_items';

    protected $guarded = ['id'];

}
