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

namespace YooKassa\Request\Payments\PaymentOrderData;

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PaymentOrderRecipientBank.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property string $name Название банка получателя.
 * @property string $bic БИК банка получателя.
 * @property string $account Счет получателя в банке.
 * @property string $correspondent_account Корреспондентский счет банка получателя.
 */
class PaymentOrderRecipientBank extends AbstractObject
{
    /**
     * Название банка получателя.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    private ?string $_name = null;

    /**
     * БИК банка получателя.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Regex("/^[0-9]{9}$/")]
    private ?string $_bic = null;

    /**
     * Счет получателя в банке.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    private ?string $_account = null;

    /**
     * Корреспондентский счет банка получателя.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    private ?string $_correspondent_account = null;

    /**
     * Возвращает name.
     *
     * @return string|null
     */
    public function getName(): ?string
    {
        return $this->_name;
    }

    /**
     * Устанавливает name.
     *
     * @param string|null $name Название банка получателя.
     *
     * @return self
     */
    public function setName(?string $name = null): self
    {
        $this->_name = $this->validatePropertyValue('_name', $name);
        return $this;
    }

    /**
     * Возвращает bic.
     *
     * @return string|null
     */
    public function getBic(): ?string
    {
        return $this->_bic;
    }

    /**
     * Устанавливает bic.
     *
     * @param string|null $bic БИК банка получателя.
     *
     * @return self
     */
    public function setBic(?string $bic = null): self
    {
        $this->_bic = $this->validatePropertyValue('_bic', $bic);
        return $this;
    }

    /**
     * Возвращает account.
     *
     * @return string|null
     */
    public function getAccount(): ?string
    {
        return $this->_account;
    }

    /**
     * Устанавливает account.
     *
     * @param string|null $account Счет получателя в банке.
     *
     * @return self
     */
    public function setAccount(?string $account = null): self
    {
        $this->_account = $this->validatePropertyValue('_account', $account);
        return $this;
    }

    /**
     * Возвращает correspondent_account.
     *
     * @return string|null
     */
    public function getCorrespondentAccount(): ?string
    {
        return $this->_correspondent_account;
    }

    /**
     * Устанавливает correspondent_account.
     *
     * @param string|null $correspondent_account Корреспондентский счет банка получателя.
     *
     * @return self
     */
    public function setCorrespondentAccount(?string $correspondent_account = null): self
    {
        $this->_correspondent_account = $this->validatePropertyValue('_correspondent_account', $correspondent_account);
        return $this;
    }
}
