<?php
namespace Modules\Treasury\Models; use Illuminate\Database\Eloquent\Model;
class BankTransaction extends Model{protected $fillable=['bank_account_id','date','description','amount','direction','reference','status'];}