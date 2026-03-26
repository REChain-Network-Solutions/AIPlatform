<?php

use Illuminate\Support\Facades\Route;

// Auto-bridge all donor route files under Routes/Donor into admin/compliance/donor/*
foreach (glob(__DIR__ . '/Donor/*.php') as $donorRoute) {
    require $donorRoute;
}
