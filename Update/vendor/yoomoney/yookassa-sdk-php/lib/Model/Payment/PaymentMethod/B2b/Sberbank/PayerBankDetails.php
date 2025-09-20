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

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель B2bSberbankPayerBankDetails.
 *
 * Банковские реквизиты плательщика (юридического лица или ИП).
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $fullName Полное наименование организации
 * @property string $full_name Полное наименование организации
 * @property string $shortName Сокращенное наименование организации
 * @property string $short_name Сокращенное наименование организации
 * @property string $address Адрес организации
 * @property string $inn ИНН организации
 * @property string $kpp КПП организации
 * @property string $bankName Наименование банка организации
 * @property string $bank_name Наименование банка организации
 * @property string $bankBranch Отделение банка организации
 * @property string $bank_branch Отделение банка организации
 * @property string $bankBik БИК банка организации
 * @property string $bank_bik БИК банка организации
 * @property string $account Номер счета организации
 */
class PayerBankDetails extends AbstractObject implements PayerBankDetailsInterface
{
    /**
     * Полное наименование организации.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: 800)]
    private ?string $_full_name = null;

    /**
     * Сокращенное наименование организации.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: 160)]
    private ?string $_short_name = null;

    /**
     * Адрес организации.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: 500)]
    private ?string $_address = null;

    /**
     * Индивидуальный налоговый номер (ИНН) организации.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Regex('/^\d{10}|\d{12}$/')]
    private ?string $_inn = null;

    /**
     * Наименование банка организации.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: 350)]
    #[Assert\Length(min: 1)]
    private ?string $_bank_name = null;

    /**
     * Отделение банка организации.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: 140)]
    #[Assert\Length(min: 1)]
    private ?string $_bank_branch = null;

    /**
     * Банковский идентификационный код (БИК) банка организации.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Regex('/^\d{9}$/')]
    private ?string $_bank_bik = null;

    /**
     * Номер счета организации.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Regex('/^\d{20}$/')]
    private ?string $_account = null;

    /**
     * Код причины постановки на учет (КПП) организации.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Regex('/^\d{9}$/')]
    private ?string $_kpp = null;

    /**
     * Возвращает полное наименование организации.
     *
     * @return string|null Полное наименование организации
     */
    public function getFullName(): ?string
    {
        return $this->_full_name;
    }

    /**
     * Устанавливает полное наименование организации.
     *
     * @param string|null $full_name Полное наименование организации
     */
    public function setFullName(?string $full_name): self
    {
        $this->_full_name = $this->validatePropertyValue('_full_name', $full_name);
        return $this;
    }

    /**
     * Возвращает сокращенное наименование организации.
     *
     * @return string|null Сокращенное наименование организации
     */
    public function getShortName(): ?string
    {
        return $this->_short_name;
    }

    /**
     * Устанавливает сокращенное наименование организации.
     *
     * @param string|null $short_name Сокращенное наименование организации
     */
    public function setShortName(?string $short_name): self
    {
        $this->_short_name = $this->validatePropertyValue('_short_name', $short_name);
        return $this;
    }

    /**
     * Возвращает адрес организации.
     *
     * @return string|null Адрес организации
     */
    public function getAddress(): ?string
    {
        return $this->_address;
    }

    /**
     * Устанавливает адрес организации.
     *
     * @param string|null $address Адрес организации
     */
    public function setAddress(?string $address): self
    {
        $this->_address = $this->validatePropertyValue('_address', $address);
        return $this;
    }

    /**
     * Возвращает ИНН организации.
     *
     * @return string|null ИНН организации
     */
    public function getInn(): ?string
    {
        return $this->_inn;
    }

    /**
     * Устанавливает ИНН организации.
     *
     * @param string|null $inn ИНН организации
     */
    public function setInn(?string $inn): self
    {
        $this->_inn = $this->validatePropertyValue('_inn', $inn);
        return $this;
    }

    /**
     * Возвращает КПП организации.
     *
     * @return string|null КПП организации
     */
    public function getKpp(): ?string
    {
        return $this->_kpp;
    }

    /**
     * Устанавливает КПП организации.
     *
     * @param string|null $kpp КПП организации
     */
    public function setKpp(?string $kpp): self
    {
        $this->_kpp = $this->validatePropertyValue('_kpp', $kpp);
        return $this;
    }

    /**
     * Возвращает наименование банка организации.
     *
     * @return string|null Наименование банка организации
     */
    public function getBankName(): ?string
    {
        return $this->_bank_name;
    }

    /**
     * Устанавливает наименование банка организации.
     *
     * @param string|null $bank_name Наименование банка организации
     */
    public function setBankName(?string $bank_name): self
    {
        $this->_bank_name = $this->validatePropertyValue('_bank_name', $bank_name);
        return $this;
    }

    /**
     * Возвращает отделение банка организации.
     *
     * @return string|null Отделение банка организации
     */
    public function getBankBranch(): ?string
    {
        return $this->_bank_branch;
    }

    /**
     * Устанавливает отделение банка организации.
     *
     * @param string|null $bank_branch Отделение банка организации
     */
    public function setBankBranch(?string $bank_branch): self
    {
        $this->_bank_branch = $this->validatePropertyValue('_bank_branch', $bank_branch);
        return $this;
    }

    /**
     * Возвращает БИК банка организации.
     *
     * @return string|null БИК банка организации
     */
    public function getBankBik(): ?string
    {
        return $this->_bank_bik;
    }

    /**
     * Устанавливает БИК банка организации.
     *
     * @param string|null $bank_bik БИК банка организации
     */
    public function setBankBik(?string $bank_bik): self
    {
        $this->_bank_bik = $this->validatePropertyValue('_bank_bik', $bank_bik);
        return $this;
    }

    /**
     * Возвращает номер счета организации.
     *
     * @return string|null Номер счета организации
     */
    public function getAccount(): ?string
    {
        return $this->_account;
    }

    /**
     * Устанавливает номер счета организации.
     *
     * @param string|null $account Номер счета организации
     */
    public function setAccount(?string $account): self
    {
        $this->_account = $this->validatePropertyValue('_account', $account);
        return $this;
    }
}
