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

namespace YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank;

/**
 * Interface PayerBankDetailsInterface.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $fullName Полное наименование организации
 * @property string $shortName Сокращенное наименование организации
 * @property string $address Адрес организации
 * @property string $inn ИНН организации
 * @property string $kpp КПП организации
 * @property string $bankName Наименование банка организации
 * @property string $bankBranch Отделение банка организации
 * @property string $bankBik БИК банка организации
 * @property string $account Номер счета организации
 */
interface PayerBankDetailsInterface
{
    /**
     * Возвращает полное наименование организации.
     *
     * @return string|null Полное наименование организации
     */
    public function getFullName(): ?string;

    /**
     * Возвращает сокращенное наименование организации.
     *
     * @return string|null Сокращенное наименование организации
     */
    public function getShortName(): ?string;

    /**
     * Возвращает адрес организации.
     *
     * @return string|null Адрес организации
     */
    public function getAddress(): ?string;

    /**
     * Возвращает ИНН организации.
     *
     * @return string|null ИНН организации
     */
    public function getInn(): ?string;

    /**
     * Возвращает КПП организации.
     *
     * @return string|null КПП организации
     */
    public function getKpp(): ?string;

    /**
     * Возвращает наименование банка организации.
     *
     * @return string|null Наименование банка организации
     */
    public function getBankName(): ?string;

    /**
     * Возвращает отделение банка организации.
     *
     * @return string|null Отделение банка организации
     */
    public function getBankBranch(): ?string;

    /**
     * Возвращает БИК банка организации.
     *
     * @return string|null БИК банка организации
     */
    public function getBankBik(): ?string;

    /**
     * Возвращает номер счета организации.
     *
     * @return string|null Номер счета организации
     */
    public function getAccount(): ?string;
}
