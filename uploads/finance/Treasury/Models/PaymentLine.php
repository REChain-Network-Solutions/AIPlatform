<?php
namespace Modules\Treasury\Models;
use Illuminate\Database\Eloquent\Model;
class PaymentLine extends Model {
  protected $fillable = ['payment_run_id','beneficiary','amount','reference','status'];
  public function run(){ return $this->belongsTo(PaymentRun::class, 'payment_run_id'); }
}
