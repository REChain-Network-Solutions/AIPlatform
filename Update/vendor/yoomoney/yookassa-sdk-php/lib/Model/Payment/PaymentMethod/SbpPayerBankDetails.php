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

namespace YooKassa\Model\Payment\PaymentMethod;

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель SbpPayerBankDetails.
 *
 * Реквизиты счета, который использовался для оплаты.
 * Обязательный параметр для платежей в статусе ~`succeeded`. В остальных случаях может отсутствовать.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $bank_id Идентификатор банка или платежного сервиса в СБП (НСПК).
 * @property string $bic Банковский идентификационный код (БИК) банка или платежного сервиса.
*/
class SbpPayerBankDetails extends AbstractObject
{
    /**
     * Идентификатор банка или платежного сервиса в СБП (НСПК).
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: 12)]
    #[Assert\Regex("/[a-zA-Z0-9]{12}/")]
    private ?string $_bank_id = null;

    /**
     * Банковский идентификационный код (БИК) банка или платежного сервиса.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    private ?string $_bic = null;

    /**
     * Возвращает bank_id.
     *
     * @return string|null
     */
    public function getBankId(): ?string
    {
        return $this->_bank_id;
    }

    /**
     * Устанавливает bank_id.
     *
     * @param string|null $bank_id Идентификатор банка или платежного сервиса в СБП (НСПК).
     *
     * @return self
     */
    public function setBankId(?string $bank_id = null): self
    {
        $this->_bank_id = $this->validatePropertyValue('_bank_id', $bank_id);
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
     * @param string|null $bic
     *
     * @return self
     */
    public function setBic(?string $bic = null): self
    {
        $this->_bic = $this->validatePropertyValue('_bic', $bic);
        return $this;
    }

}

