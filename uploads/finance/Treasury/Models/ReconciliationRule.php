<?php
namespace Modules\Treasury\Models;
use Illuminate\Database\Eloquent\Model;
class ReconciliationRule extends Model {
  protected $fillable = ['pattern','account_code','direction']; // simple mapping
}
