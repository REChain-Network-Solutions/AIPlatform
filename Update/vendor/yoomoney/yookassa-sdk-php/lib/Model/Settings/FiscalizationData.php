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

namespace YooKassa\Model\Settings;

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель FiscalizationData.
 *
 * Настройки магазина для [отправки чеков в налоговую](/developers/payment-acceptance/receipts/basics).
 *
 * Присутствует, если вы запрашивали настройки магазина, и этот магазин использует решения ЮKassa для отправки чеков.
 * Отсутствует, если магазин еще не включал отправку чеков через ЮKassa.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property bool $enabled В настройках магазина включена отправка чеков.
 * @property string $provider
 */
class FiscalizationData extends AbstractObject
{
    /**
     * В настройках магазина включена отправка чеков. Возможные значения: `true` — магазин отправляет данные для чеков через ЮKassa; `false` — магазин выключил отправку чеков через ЮKassa.
     *
     * @var bool|null
     */
    #[Assert\NotNull]
    #[Assert\Type('bool')]
    private ?bool $_enabled = null;

    /**
     * Решение ЮKassa, которое магазин использует для отправки чеков.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    private ?string $_provider = null;

    /**
     * Возвращает enabled.
     *
     * @return bool|null
     */
    public function getEnabled(): ?bool
    {
        return $this->_enabled;
    }

    /**
     * Устанавливает enabled.
     *
     * @param bool|null $enabled В настройках магазина включена отправка чеков.
     *
     * @return self
     */
    public function setEnabled(?bool $enabled = null): self
    {
        $this->_enabled = $this->validatePropertyValue('_enabled', $enabled);
        return $this;
    }

    /**
     * Возвращает provider.
     *
     * @return string|null
     */
    public function getProvider(): ?string
    {
        return $this->_provider;
    }

    /**
     * Устанавливает provider.
     *
     * @param string|null $provider
     *
     * @return self
     */
    public function setProvider(?string $provider = null): self
    {
        $this->_provider = $this->validatePropertyValue('_provider', $provider);
        return $this;
    }

}

