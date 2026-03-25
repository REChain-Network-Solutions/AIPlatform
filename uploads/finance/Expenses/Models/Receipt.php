<?php
namespace Modules\Expenses\Models;
use Illuminate\Database\Eloquent\Model;
class Receipt extends Model {
  protected $table = 'expense_receipts';
  protected $fillable = ['expense_id','path','mime','size','ocr_text'];
}
