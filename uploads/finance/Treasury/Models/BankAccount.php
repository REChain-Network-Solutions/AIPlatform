<?php
namespace Modules\Treasury\Models; use Illuminate\Database\Eloquent\Model;
class BankAccount extends Model{protected $fillable=['name','iban','bsb','account_number','currency','opening_balance'];}