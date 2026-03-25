<?php
namespace Modules\Treasury\Models;
use Illuminate\Database\Eloquent\Model;
class PaymentRun extends Model {
  protected $fillable = ['scheduled_on','status','bank_account_id','posted_journal_id'];
  public function lines(){ return $this->hasMany(PaymentLine::class); }
}
