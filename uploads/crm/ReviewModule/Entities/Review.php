<?php

namespace Modules\ReviewModule\Entities;

use App\Traits\HasUuid;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Modules\BookingModule\Entities\Booking;
use Modules\ProviderManagement\Entities\Provider;
use Modules\ServiceManagement\Entities\Service;
use Modules\UserManagement\Entities\User;
use Modules\ReviewModule\Traits\CompanyScoped;

class Review extends Model
{
    use CompanyScoped;
    use HasFactory;
    use HasUuid;

    protected $casts = [
        'review_rating' => 'integer',
        'review_images' => 'array',
        'is_active' => 'integer',
    ];

    protected $fillable = [];

    public function booking(): \Illuminate\Database\Eloquent\Relations\BelongsTo
    {
        return $this->belongsTo(Booking::class);
    }

    public function provider(): \Illuminate\Database\Eloquent\Relations\BelongsTo
    {
        return $this->belongsTo(Provider::class);
    }

    public function service(): \Illuminate\Database\Eloquent\Relations\BelongsTo
    {
        return $this->belongsTo(Service::class);
    }

    public function customer(): \Illuminate\Database\Eloquent\Relations\BelongsTo
    {
        return $this->belongsTo(User::class,'customer_id');
    }

    public function reviewReply(): \Illuminate\Database\Eloquent\Relations\HasOne
    {
        return $this->hasOne(ReviewReply::class,'review_id');
    }

    protected function scopeOfStatus($query, $status)
    {
        $query->where('is_active', $status);
    }
}