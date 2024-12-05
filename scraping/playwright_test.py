import { test, expect } from '@playwright/test';

test('test', async ({ page }) => {
  await page.goto('https://www.bossard.com/eshop/ch-de/');
  await page.getByRole('button', { name: 'Zustimmen' }).click();
  await page.getByRole('link', { name: 'Norm- und Standard' }).click();
  await page.getByRole('link', { name: 'Klemm- und Positionierelemente Klemm- und Positionierelemente 40 Produkte' }).click();
  await page.goto('https://www.bossard.com/eshop/ch-de/');
  await page.getByRole('link', { name: 'Einpresstechnik' }).click({
    button: 'right'
  });
  await page.getByRole('link', { name: 'Einpresstechnik' }).click();
  await page.getByRole('link', { name: 'PEM® S/SS/H BN 28429 PEM® S/' }).click({
    button: 'right'
  });
  await page.goto('https://www.bossard.com/eshop/ch-de/');
  await page.getByRole('link', { name: 'Clip-Befestigungen Clip-' }).click();
  await page.getByRole('banner').getByRole('navigation').getByRole('link', { name: 'Produkte' }).click();
});