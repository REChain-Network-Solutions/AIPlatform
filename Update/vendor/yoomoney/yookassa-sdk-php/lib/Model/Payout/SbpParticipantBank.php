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

namespace YooKassa\Model\Payout;

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель SbpParticipantBank.
 *
 * Объект участника СБП (Системы быстрых платежей ЦБ РФ) — актуальная информация о банке или платежном сервисе, подключенном к СБП.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property string $bank_id Идентификатор банка или платежного сервиса в СБП.
 * @property string $bankId Идентификатор банка или платежного сервиса в СБП.
 * @property string $name Название банка или платежного сервиса в СБП.
 * @property string $bic Банковский идентификационный код (БИК) банка или платежного сервиса.
 */
class SbpParticipantBank extends AbstractObject
{
    /** @var int Максимальная длина строки id банка. */
    public const MAX_LENGTH_BANK_ID = 12;

    /** @var int Минимальная длина строки название банка. */
    public const MAX_LENGTH_NAME = 128;

    /**
     * Идентификатор банка или платежного сервиса в СБП.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_LENGTH_BANK_ID)]
    private ?string $_bank_id = null;

    /**
     * Название банка или платежного сервиса в СБП.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_LENGTH_NAME)]
    private ?string $_name = null;

    /**
     * Банковский идентификационный код (БИК) банка или платежного сервиса.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Regex("/^\d{9}$/")]
    private ?string $_bic = null;

    /**
     * Возвращает идентификатор банка или платежного сервиса в СБП.
     *
     * @return string|null Идентификатор банка или платежного сервиса в СБП
     */
    public function getBankId(): ?string
    {
        return $this->_bank_id;
    }

    /**
     * Устанавливает идентификатор банка или платежного сервиса в СБП.
     *
     * @param string|null $bank_id Идентификатор банка или платежного сервиса в СБП
     *
     * @return self
     */
    public function setBankId(?string $bank_id = null): self
    {
        $this->_bank_id = $this->validatePropertyValue('_bank_id', $bank_id);
        return $this;
    }

    /**
     * Возвращает название банка или платежного сервиса в СБП.
     *
     * @return string|null Название банка или платежного сервиса в СБП
     */
    public function getName(): ?string
    {
        return $this->_name;
    }

    /**
     * Устанавливает название банка или платежного сервиса в СБП.
     *
     * @param string|null $name Название банка или платежного сервиса в СБП
     *
     * @return self
     */
    public function setName(?string $name = null): self
    {
        $this->_name = $this->validatePropertyValue('_name', $name);
        return $this;
    }

    /**
     * Возвращает банковский идентификационный код.
     *
     * @return string|null Банковский идентификационный код (БИК) банка или платежного сервиса
     */
    public function getBic(): ?string
    {
        return $this->_bic;
    }

    /**
     * Устанавливает банковский идентификационный код.
     *
     * @param string|null $bic Банковский идентификационный код (БИК) банка или платежного сервиса
     *
     * @return self
     */
    public function setBic(?string $bic = null): self
    {
        $this->_bic = $this->validatePropertyValue('_bic', $bic);
        return $this;
    }
}
