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

namespace YooKassa\Model\Deal;

use YooKassa\Common\AbstractObject;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\CurrencyCode;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель DealBalanceAmount.
 *
 * Баланс сделки.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property int $value Сумма в выбранной валюте.
 * @property string $currency Трехбуквенный код валюты в формате ISO-4217. Пример: RUB.
 */
class DealBalanceAmount extends AbstractObject implements AmountInterface
{
    /**
     * Сумма в выбранной валюте.
     *
     * @var int|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('int')]
    private ?int $_value = 0;

    /**
     * Трехбуквенный код валюты в формате ISO-4217. Пример: RUB.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Choice(callback: [CurrencyCode::class, 'getValidValues'])]
    #[Assert\Type('string')]
    private ?string $_currency = CurrencyCode::RUB;

    /**
     * MonetaryAmount constructor.
     *
     * @param null|array|numeric $value Сумма
     * @param null|string $currency Код валюты
     */
    public function __construct(mixed $value = null, ?string $currency = null)
    {
        if (is_array($value)) {
            parent::__construct($value);
        } else {
            parent::__construct();
            if (null !== $value) {
                $this->setValue($value);
            }
            if (null !== $currency) {
                $this->setCurrency($currency);
            }
        }
    }

    /**
     * Возвращает значение суммы.
     *
     * @return string Сумма
     */
    public function getValue(): string
    {
        $negative = ($this->_value < 0 ? '-' : '');
        $mod = abs($this->_value);
        if ($mod < 10) {
            return $negative . '0.0' . $mod;
        }
        if ($mod < 100) {
            return $negative . '0.' . $mod;
        }

        return $negative . substr($mod, 0, -2) . '.' . substr($mod, -2);
    }

    /**
     * Устанавливает сумму.
     *
     * @param numeric|string $value Сумма
     *
     * @return self
     */
    public function setValue(mixed $value): self
    {
        if (is_numeric($value)) {
            $value = (int) round($value * 100.0);
        }
        $this->_value = $this->validatePropertyValue('_value', $value);
        return $this;
    }

    /**
     * Возвращает сумму в копейках в виде целого числа.
     *
     * @return int Сумма в копейках/центах
     */
    public function getIntegerValue(): int
    {
        return $this->_value;
    }

    /**
     * Возвращает валюту.
     *
     * @return string Код валюты
     */
    public function getCurrency(): string
    {
        return $this->_currency;
    }

    /**
     * Устанавливает код валюты.
     *
     * @param string|null $currency Код валюты
     *
     * @return self
     */
    public function setCurrency(?string $currency): self
    {
        $this->_currency = $this->validatePropertyValue('_currency', $currency);
        return $this;
    }

    /**
     * @return array
     */
    public function jsonSerialize(): array
    {
        return [
            'value' => number_format($this->_value / 100.0, 2, '.', ''),
            'currency' => $this->_currency,
        ];
    }
}
