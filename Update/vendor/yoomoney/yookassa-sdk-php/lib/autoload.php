<?php

/*
 * The MIT License
 *
 * Copyright (c) 2025 "YooMoney", NBСO LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
const YOOKASSA_SDK_ROOT_PATH = __DIR__;
const YOOKASSA_SDK_PSR_LOG_PATH = __DIR__ . '/../vendor/psr/log/Psr/Log';

function yookassaSdkLoadClass($className): void
{
    if (0 === strncmp('YooKassa', $className, 8)) {
        $path = YOOKASSA_SDK_ROOT_PATH;
        $length = 8;
    } elseif (0 === strncmp('Psr\Log', $className, 7)) {
        $path = YOOKASSA_SDK_PSR_LOG_PATH;
        $length = 7;
    } else {
        return;
    }
    $path .= str_replace('\\', '/', substr($className, $length)) . '.php';
    if (file_exists($path)) {
        require $path;
    }
}

spl_autoload_register('yookassaSdkLoadClass');
