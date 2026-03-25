<?php

/**
 * Author: Emmanuel Paul Mnzava
 * Twitter: @epmnzava
 * Github:https://github.com/dbrax/income-expense
 * Email: epmnzava@gmail.com
 * 
 */

namespace  Epmnzava\IncomeExpense\Models;

use Illuminate\Database\Eloquent\Model;
use Epmnzava\IncomeExpense\Models\ExpenseCategory;

class Expense extends Model
{

    protected $guarded = [];
    protected $table = "expense";


       public function category()
    {
        return $this->belongsTo(ExpenseCategory::class,'expense_category');
    }
}
