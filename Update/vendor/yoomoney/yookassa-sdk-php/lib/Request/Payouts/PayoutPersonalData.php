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

namespace YooKassa\Request\Payouts;

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PayoutPersonalData.
 *
 * Персональные данные получателя выплаты. Только для обычных выплат.
 * Необходимо передавать в этих сценариях:
 *  * [выплаты с проверкой получателя](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-check);
 *  * [выплаты с передачей данных получателя для выписок из реестра](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-data-send).
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property string $id Идентификатор персональных данных, сохраненных в ЮKassa.
 */
class PayoutPersonalData extends AbstractObject
{
    /** @var int Максимальная длина идентификатора персональных данных */
    public const MAX_LENGTH_ID = 50;

    /** @var int Максимальная длина идентификатора персональных данных */
    public const MIN_LENGTH_ID = 36;

    /**
     * Идентификатор персональных данных, сохраненных в ЮKassa.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_LENGTH_ID)]
    #[Assert\Length(min: self::MIN_LENGTH_ID)]
    private ?string $_id = null;

    /**
     * Возвращает идентификатор персональных данных.
     *
     * @return string|null Идентификатор персональных данных
     */
    public function getId(): ?string
    {
        return $this->_id;
    }

    /**
     * Устанавливает идентификатор персональных данных.
     *
     * @param string|null $id Идентификатор персональных данных, сохраненных в ЮKassa
     *
     * @return self
     */
    public function setId(?string $id = null): self
    {
        $this->_id = $this->validatePropertyValue('_id', $id);
        return $this;
    }
}
