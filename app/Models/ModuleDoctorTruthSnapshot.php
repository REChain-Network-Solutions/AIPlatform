<?php
namespace App\Models;
use Illuminate\Database\Eloquent\Model;
class ModuleDoctorTruthSnapshot extends Model {
    protected $guarded = [];
    protected $casts = ['payload' => 'array'];
}
